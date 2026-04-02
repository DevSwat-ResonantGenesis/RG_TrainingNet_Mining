"""
Pipeline Coordinator — Microbatch Scheduling for Distributed Pipeline Parallelism
===================================================================================

Manages the execution schedule for pipeline-parallel training across miners.
Supports two scheduling strategies:

  1. GPipe: All forward microbatches first, then all backward (simpler, more memory)
  2. 1F1B: Interleaved forward/backward (less memory, preferred for large models)

Scales from 2-stage pipelines (Phase 5) to 1000+ stage pipelines (future).

The coordinator runs on the Mining Service and tells each miner WHEN to:
  - Execute a forward pass on a microbatch
  - Wait for upstream activations
  - Execute a backward pass
  - Send activations/gradients downstream/upstream
  - Submit local gradients to the sharded parameter server
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.pipeline")


# ══════════════════════════════════════════════════════════════
# SCHEDULE TYPES
# ══════════════════════════════════════════════════════════════

class ScheduleStrategy(str, Enum):
    GPIPE = "gpipe"     # All forwards, then all backwards (simpler)
    ONE_F_ONE_B = "1f1b"  # Interleaved (memory-efficient, preferred)
    CHIMERA = "chimera"   # Bidirectional (advanced, future)


class MicrobatchAction(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    SUBMIT_GRADIENTS = "submit_gradients"
    SYNC = "sync"


@dataclass
class ScheduleStep:
    """One step in the pipeline schedule for a specific stage."""
    step_index: int
    stage_index: int
    action: MicrobatchAction
    microbatch_index: int = -1       # Which microbatch (-1 for non-F/B actions)
    clock_tick: int = 0              # Global clock tick for ordering
    depends_on: Optional[int] = None  # Step index this depends on (activation from upstream)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "stage_index": self.stage_index,
            "action": self.action.value,
            "microbatch_index": self.microbatch_index,
            "clock_tick": self.clock_tick,
            "depends_on": self.depends_on,
        }


@dataclass
class PipelineSchedule:
    """
    Complete execution schedule for one training step across all pipeline stages.
    
    Contains ordered steps for each stage. The coordinator distributes
    each stage's steps to the corresponding miner.
    """
    schedule_id: str = field(default_factory=lambda: f"sched-{uuid4().hex[:8]}")
    pipeline_group_id: str = ""
    strategy: ScheduleStrategy = ScheduleStrategy.ONE_F_ONE_B
    num_stages: int = 1
    num_microbatches: int = 4
    # stage_index → ordered list of steps
    stage_steps: Dict[int, List[ScheduleStep]] = field(default_factory=dict)
    total_ticks: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def efficiency(self) -> float:
        """Pipeline efficiency: useful compute ticks / total ticks."""
        if self.total_ticks == 0:
            return 0.0
        useful = self.num_microbatches * 2  # forward + backward per microbatch
        total_per_stage = self.total_ticks
        return useful / total_per_stage if total_per_stage > 0 else 0.0

    @property
    def bubble_fraction(self) -> float:
        """Fraction of time spent in pipeline bubbles (idle)."""
        return 1.0 - self.efficiency

    def get_steps_for_stage(self, stage_index: int) -> List[ScheduleStep]:
        """Get ordered execution steps for a specific stage."""
        return self.stage_steps.get(stage_index, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_id": self.schedule_id,
            "pipeline_group_id": self.pipeline_group_id,
            "strategy": self.strategy.value,
            "num_stages": self.num_stages,
            "num_microbatches": self.num_microbatches,
            "total_ticks": self.total_ticks,
            "efficiency": round(self.efficiency, 4),
            "bubble_fraction": round(self.bubble_fraction, 4),
            "stage_steps": {
                k: [s.to_dict() for s in v]
                for k, v in self.stage_steps.items()
            },
        }


# ══════════════════════════════════════════════════════════════
# SCHEDULE GENERATORS
# ══════════════════════════════════════════════════════════════

def generate_gpipe_schedule(num_stages: int, num_microbatches: int) -> PipelineSchedule:
    """
    Generate a GPipe schedule.
    
    All forward passes first (filling pipeline), then all backward passes.
    
    Example with 4 stages, 4 microbatches (F=forward, B=backward):
    
    Tick:    0  1  2  3  4  5  6  7  8  9  10
    Stage 0: F0 F1 F2 F3 -- -- -- B3 B2 B1 B0
    Stage 1: -- F0 F1 F2 F3 -- -- -- B3 B2 B1
    Stage 2: -- -- F0 F1 F2 F3 -- -- -- B3 B2
    Stage 3: -- -- -- F0 F1 F2 F3 B3 B2 B1 B0  (loss computed here)
    
    Total ticks = 2 * (num_microbatches + num_stages - 1)
    Efficiency = 2M / (2M + 2(S-1)) where M=microbatches, S=stages
    """
    schedule = PipelineSchedule(
        strategy=ScheduleStrategy.GPIPE,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
    )

    step_counter = 0
    total_ticks = 2 * num_microbatches + 2 * (num_stages - 1)
    schedule.total_ticks = total_ticks

    for stage in range(num_stages):
        steps = []

        # Forward phase: staggered start
        for mb in range(num_microbatches):
            tick = stage + mb
            step = ScheduleStep(
                step_index=step_counter,
                stage_index=stage,
                action=MicrobatchAction.FORWARD,
                microbatch_index=mb,
                clock_tick=tick,
            )
            # Depends on upstream forward of same microbatch
            if stage > 0:
                step.depends_on = _find_step_index(
                    schedule, stage - 1, MicrobatchAction.FORWARD, mb
                )
            steps.append(step)
            step_counter += 1

        # Backward phase: reverse order, staggered
        for mb in reversed(range(num_microbatches)):
            tick = num_microbatches + (num_stages - 1) + (num_stages - 1 - stage) + (num_microbatches - 1 - mb)
            step = ScheduleStep(
                step_index=step_counter,
                stage_index=stage,
                action=MicrobatchAction.BACKWARD,
                microbatch_index=mb,
                clock_tick=tick,
            )
            # Depends on downstream backward of same microbatch
            if stage < num_stages - 1:
                step.depends_on = _find_step_index(
                    schedule, stage + 1, MicrobatchAction.BACKWARD, mb
                )
            steps.append(step)
            step_counter += 1

        # Final: submit gradients
        steps.append(ScheduleStep(
            step_index=step_counter,
            stage_index=stage,
            action=MicrobatchAction.SUBMIT_GRADIENTS,
            clock_tick=total_ticks,
        ))
        step_counter += 1

        schedule.stage_steps[stage] = steps

    return schedule


def generate_1f1b_schedule(num_stages: int, num_microbatches: int) -> PipelineSchedule:
    """
    Generate a 1F1B (one-forward-one-backward) interleaved schedule.
    
    More memory-efficient than GPipe because it interleaves forward and
    backward passes, so fewer activations need to be held in memory.
    
    Example with 4 stages, 8 microbatches:
    
    Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
    Stage 1:    F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
    Stage 2:       F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6
    Stage 3:          F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 F7 B4 B5
    
    Warmup: first `num_stages` forward passes fill the pipeline
    Steady state: alternating 1 forward + 1 backward
    Cooldown: remaining backward passes drain the pipeline
    
    Peak activation memory: num_stages activations (vs num_microbatches for GPipe)
    """
    schedule = PipelineSchedule(
        strategy=ScheduleStrategy.ONE_F_ONE_B,
        num_stages=num_stages,
        num_microbatches=num_microbatches,
    )

    step_counter = 0

    for stage in range(num_stages):
        steps = []
        tick = stage  # Staggered start

        # Warmup: forward passes to fill the pipeline
        warmup_count = num_stages - stage - 1
        warmup_count = min(warmup_count, num_microbatches)

        forward_mb = 0
        backward_mb = 0

        # Warmup forwards
        for _ in range(warmup_count):
            if forward_mb >= num_microbatches:
                break
            step = ScheduleStep(
                step_index=step_counter,
                stage_index=stage,
                action=MicrobatchAction.FORWARD,
                microbatch_index=forward_mb,
                clock_tick=tick,
            )
            if stage > 0:
                step.depends_on = _find_step_index(
                    schedule, stage - 1, MicrobatchAction.FORWARD, forward_mb
                )
            steps.append(step)
            step_counter += 1
            forward_mb += 1
            tick += 1

        # Steady state: 1 forward + 1 backward alternating
        while forward_mb < num_microbatches:
            # Forward
            step = ScheduleStep(
                step_index=step_counter,
                stage_index=stage,
                action=MicrobatchAction.FORWARD,
                microbatch_index=forward_mb,
                clock_tick=tick,
            )
            if stage > 0:
                step.depends_on = _find_step_index(
                    schedule, stage - 1, MicrobatchAction.FORWARD, forward_mb
                )
            steps.append(step)
            step_counter += 1
            forward_mb += 1
            tick += 1

            # Backward (if we have microbatches ready)
            if backward_mb < num_microbatches:
                step = ScheduleStep(
                    step_index=step_counter,
                    stage_index=stage,
                    action=MicrobatchAction.BACKWARD,
                    microbatch_index=backward_mb,
                    clock_tick=tick,
                )
                if stage < num_stages - 1:
                    step.depends_on = _find_step_index(
                        schedule, stage + 1, MicrobatchAction.BACKWARD, backward_mb
                    )
                steps.append(step)
                step_counter += 1
                backward_mb += 1
                tick += 1

        # Cooldown: remaining backward passes
        while backward_mb < num_microbatches:
            step = ScheduleStep(
                step_index=step_counter,
                stage_index=stage,
                action=MicrobatchAction.BACKWARD,
                microbatch_index=backward_mb,
                clock_tick=tick,
            )
            if stage < num_stages - 1:
                step.depends_on = _find_step_index(
                    schedule, stage + 1, MicrobatchAction.BACKWARD, backward_mb
                )
            steps.append(step)
            step_counter += 1
            backward_mb += 1
            tick += 1

        # Submit gradients after all backward passes
        steps.append(ScheduleStep(
            step_index=step_counter,
            stage_index=stage,
            action=MicrobatchAction.SUBMIT_GRADIENTS,
            clock_tick=tick,
        ))
        step_counter += 1

        schedule.stage_steps[stage] = steps
        schedule.total_ticks = max(schedule.total_ticks, tick + 1)

    return schedule


def _find_step_index(
    schedule: PipelineSchedule,
    stage: int,
    action: MicrobatchAction,
    microbatch: int,
) -> Optional[int]:
    """Find the step index for a specific stage/action/microbatch combination."""
    steps = schedule.stage_steps.get(stage, [])
    for s in steps:
        if s.action == action and s.microbatch_index == microbatch:
            return s.step_index
    return None


def choose_schedule(
    num_stages: int,
    num_microbatches: int,
    prefer_memory_efficient: bool = True,
) -> PipelineSchedule:
    """
    Auto-select the best schedule strategy.
    
    Rules:
    - 1 stage: no schedule needed (just forward/backward)
    - 2-4 stages with few microbatches: GPipe (simpler)
    - 5+ stages or many microbatches: 1F1B (memory-efficient)
    - prefer_memory_efficient=True: always use 1F1B when possible
    """
    if num_stages <= 1:
        # Trivial schedule: one forward, one backward
        schedule = PipelineSchedule(
            num_stages=1,
            num_microbatches=num_microbatches,
            total_ticks=2,
        )
        steps = []
        for mb in range(num_microbatches):
            steps.append(ScheduleStep(
                step_index=mb,
                stage_index=0,
                action=MicrobatchAction.FORWARD,
                microbatch_index=mb,
                clock_tick=0,
            ))
        for mb in range(num_microbatches):
            steps.append(ScheduleStep(
                step_index=num_microbatches + mb,
                stage_index=0,
                action=MicrobatchAction.BACKWARD,
                microbatch_index=mb,
                clock_tick=1,
            ))
        steps.append(ScheduleStep(
            step_index=2 * num_microbatches,
            stage_index=0,
            action=MicrobatchAction.SUBMIT_GRADIENTS,
            clock_tick=2,
        ))
        schedule.stage_steps[0] = steps
        return schedule

    # Ensure enough microbatches for efficiency
    num_microbatches = max(num_microbatches, num_stages)

    if prefer_memory_efficient or num_stages >= 5:
        return generate_1f1b_schedule(num_stages, num_microbatches)
    return generate_gpipe_schedule(num_stages, num_microbatches)


# ══════════════════════════════════════════════════════════════
# PIPELINE EXECUTION COORDINATOR
# ══════════════════════════════════════════════════════════════

@dataclass
class MicrobatchState:
    """Tracks the state of a single microbatch through the pipeline."""
    microbatch_index: int
    data_hash: str = ""                      # Hash of input data for verification
    forward_complete: Dict[int, bool] = field(default_factory=dict)   # stage → done
    backward_complete: Dict[int, bool] = field(default_factory=dict)
    forward_start_time: Dict[int, float] = field(default_factory=dict)
    forward_end_time: Dict[int, float] = field(default_factory=dict)
    backward_start_time: Dict[int, float] = field(default_factory=dict)
    backward_end_time: Dict[int, float] = field(default_factory=dict)
    loss: Optional[float] = None  # Set by last stage after forward


@dataclass
class PipelineExecution:
    """
    Tracks one complete pipeline training step across all stages.
    
    Created by the PipelineCoordinator when a training step begins.
    Updated as miners report forward/backward completion.
    """
    execution_id: str = field(default_factory=lambda: f"exec-{uuid4().hex[:8]}")
    pipeline_group_id: str = ""
    schedule: Optional[PipelineSchedule] = None
    global_step: int = 0
    status: str = "pending"  # pending, running, complete, failed
    microbatches: Dict[int, MicrobatchState] = field(default_factory=dict)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    # Per-stage gradient submission status
    gradients_submitted: Dict[int, bool] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """All microbatches forward+backward done, all gradients submitted."""
        if not self.schedule:
            return False
        all_mb_done = all(
            all(mb.forward_complete.get(s, False) for s in range(self.schedule.num_stages))
            and all(mb.backward_complete.get(s, False) for s in range(self.schedule.num_stages))
            for mb in self.microbatches.values()
        )
        all_grads = all(
            self.gradients_submitted.get(s, False)
            for s in range(self.schedule.num_stages)
        )
        return all_mb_done and all_grads

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "pipeline_group_id": self.pipeline_group_id,
            "global_step": self.global_step,
            "status": self.status,
            "num_microbatches": len(self.microbatches),
            "elapsed_seconds": round(self.elapsed, 2),
            "gradients_submitted": dict(self.gradients_submitted),
            "is_complete": self.is_complete,
        }


class PipelineCoordinator:
    """
    Coordinates pipeline-parallel training execution.
    
    The coordinator:
    1. Generates schedules for pipeline groups
    2. Distributes step instructions to miners
    3. Tracks microbatch progress through the pipeline
    4. Handles timeouts and failures
    5. Triggers gradient submission after all backward passes
    
    Runs on the Mining Service. One coordinator per pipeline group.
    """

    # Timeout for a single microbatch forward/backward step
    STEP_TIMEOUT_SEC = 300  # 5 minutes
    # Max concurrent pipeline executions
    MAX_CONCURRENT_EXECUTIONS = 100

    def __init__(self, pipeline_group_id: str, num_stages: int):
        self.pipeline_group_id = pipeline_group_id
        self.num_stages = num_stages
        self.executions: Dict[str, PipelineExecution] = {}
        self.current_execution: Optional[PipelineExecution] = None
        self.global_step = 0
        self._notify_callbacks: List[Callable] = []

    def on_step_notify(self, callback: Callable):
        """Register a callback for when a miner should execute a step."""
        self._notify_callbacks.append(callback)

    async def _notify_miner(self, stage_index: int, step: ScheduleStep, execution_id: str):
        """Notify a miner to execute a schedule step."""
        msg = {
            "type": "pipeline_step",
            "execution_id": execution_id,
            "step": step.to_dict(),
            "global_step": self.global_step,
        }
        for cb in self._notify_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(stage_index, msg)
                else:
                    cb(stage_index, msg)
            except Exception as e:
                logger.error(f"Failed to notify stage {stage_index}: {e}")

    def create_schedule(
        self,
        num_microbatches: int = 8,
        strategy: Optional[ScheduleStrategy] = None,
    ) -> PipelineSchedule:
        """Create an execution schedule for this pipeline group."""
        if strategy:
            if strategy == ScheduleStrategy.GPIPE:
                return generate_gpipe_schedule(self.num_stages, num_microbatches)
            elif strategy == ScheduleStrategy.ONE_F_ONE_B:
                return generate_1f1b_schedule(self.num_stages, num_microbatches)
        return choose_schedule(self.num_stages, num_microbatches)

    async def start_training_step(
        self,
        num_microbatches: int = 8,
        strategy: Optional[ScheduleStrategy] = None,
    ) -> PipelineExecution:
        """
        Start a new training step across the pipeline.
        
        1. Generate schedule
        2. Create execution tracker
        3. Send first steps to each stage
        """
        schedule = self.create_schedule(num_microbatches, strategy)
        schedule.pipeline_group_id = self.pipeline_group_id

        execution = PipelineExecution(
            pipeline_group_id=self.pipeline_group_id,
            schedule=schedule,
            global_step=self.global_step,
            status="running",
            started_at=time.time(),
        )

        # Initialize microbatch tracking
        for mb in range(num_microbatches):
            execution.microbatches[mb] = MicrobatchState(microbatch_index=mb)

        self.executions[execution.execution_id] = execution
        self.current_execution = execution

        # Send first step to each stage
        for stage_idx in range(self.num_stages):
            steps = schedule.get_steps_for_stage(stage_idx)
            if steps:
                await self._notify_miner(stage_idx, steps[0], execution.execution_id)

        logger.info(
            f"Pipeline {self.pipeline_group_id} step {self.global_step}: "
            f"{num_microbatches} microbatches, {self.num_stages} stages, "
            f"strategy={schedule.strategy.value}, efficiency={schedule.efficiency:.1%}"
        )
        return execution

    async def report_step_complete(
        self,
        execution_id: str,
        stage_index: int,
        action: MicrobatchAction,
        microbatch_index: int,
        loss: Optional[float] = None,
    ) -> Optional[ScheduleStep]:
        """
        A miner reports completing a step. Returns the next step to execute, or None.
        
        This is the core coordination loop:
        1. Mark the step as complete
        2. Check if downstream/upstream dependencies are satisfied
        3. Return the next step for this stage
        4. If all steps done, check if execution is complete
        """
        execution = self.executions.get(execution_id)
        if not execution or not execution.schedule:
            return None

        mb_state = execution.microbatches.get(microbatch_index)
        if not mb_state:
            return None

        now = time.time()

        if action == MicrobatchAction.FORWARD:
            mb_state.forward_complete[stage_index] = True
            mb_state.forward_end_time[stage_index] = now
            if loss is not None:
                mb_state.loss = loss
        elif action == MicrobatchAction.BACKWARD:
            mb_state.backward_complete[stage_index] = True
            mb_state.backward_end_time[stage_index] = now
        elif action == MicrobatchAction.SUBMIT_GRADIENTS:
            execution.gradients_submitted[stage_index] = True

        # Find next step for this stage
        steps = execution.schedule.get_steps_for_stage(stage_index)
        current_idx = None
        for i, s in enumerate(steps):
            if s.action == action and s.microbatch_index == microbatch_index:
                current_idx = i
                break

        if current_idx is not None and current_idx + 1 < len(steps):
            next_step = steps[current_idx + 1]
            # Check if dependency is met
            if next_step.depends_on is not None:
                if not self._is_dependency_met(execution, next_step.depends_on):
                    # Dependency not met — will be triggered when upstream completes
                    return None
            await self._notify_miner(stage_index, next_step, execution_id)
            return next_step

        # Check if execution is complete
        if execution.is_complete:
            execution.status = "complete"
            execution.completed_at = time.time()
            self.global_step += 1
            logger.info(
                f"Pipeline {self.pipeline_group_id} step {execution.global_step} COMPLETE "
                f"in {execution.elapsed:.1f}s"
            )

        return None

    def _is_dependency_met(self, execution: PipelineExecution, dep_step_index: int) -> bool:
        """Check if a dependency step has been completed."""
        if not execution.schedule:
            return True
        # Find the dependency step
        for steps in execution.schedule.stage_steps.values():
            for s in steps:
                if s.step_index == dep_step_index:
                    mb = execution.microbatches.get(s.microbatch_index)
                    if not mb:
                        return False
                    if s.action == MicrobatchAction.FORWARD:
                        return mb.forward_complete.get(s.stage_index, False)
                    elif s.action == MicrobatchAction.BACKWARD:
                        return mb.backward_complete.get(s.stage_index, False)
                    return True
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        completed = [e for e in self.executions.values() if e.status == "complete"]
        avg_time = (
            sum(e.elapsed for e in completed) / len(completed)
            if completed else 0
        )
        return {
            "pipeline_group_id": self.pipeline_group_id,
            "num_stages": self.num_stages,
            "global_step": self.global_step,
            "total_executions": len(self.executions),
            "completed_executions": len(completed),
            "avg_step_time": round(avg_time, 2),
            "current_execution": self.current_execution.to_dict() if self.current_execution else None,
        }


# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_optimal_microbatches(
    num_stages: int,
    batch_size: int,
    target_efficiency: float = 0.85,
) -> Tuple[int, int]:
    """
    Compute optimal number of microbatches for a given pipeline configuration.
    
    Returns (num_microbatches, microbatch_size).
    
    For 1F1B schedule:
      efficiency ≈ M / (M + S - 1)  where M = microbatches, S = stages
      
    Solving for M given target efficiency E:
      M = (S - 1) * E / (1 - E)
      
    For 85% efficiency with 4 stages: M = 3 * 0.85 / 0.15 = 17 microbatches
    For 85% efficiency with 8 stages: M = 7 * 0.85 / 0.15 = 40 microbatches
    For 85% efficiency with 100 stages: M = 99 * 0.85 / 0.15 = 561 microbatches
    """
    if num_stages <= 1:
        return 1, batch_size

    if target_efficiency >= 1.0:
        target_efficiency = 0.99

    min_microbatches = max(
        num_stages,
        int(math.ceil((num_stages - 1) * target_efficiency / (1.0 - target_efficiency)))
    )

    # Microbatch size = batch_size / num_microbatches (round up)
    microbatch_size = max(1, batch_size // min_microbatches)
    # Adjust microbatches to evenly divide batch
    num_microbatches = max(num_stages, math.ceil(batch_size / microbatch_size))

    return num_microbatches, microbatch_size


def estimate_pipeline_throughput(
    num_stages: int,
    num_microbatches: int,
    time_per_microbatch_sec: float,
    strategy: ScheduleStrategy = ScheduleStrategy.ONE_F_ONE_B,
) -> Dict[str, float]:
    """
    Estimate pipeline throughput metrics.
    
    Returns dict with:
    - samples_per_second
    - efficiency
    - bubble_time_fraction
    - total_step_time
    """
    if strategy == ScheduleStrategy.ONE_F_ONE_B:
        # 1F1B: warmup + steady + cooldown
        warmup_ticks = num_stages - 1
        steady_ticks = 2 * (num_microbatches - num_stages + 1)
        cooldown_ticks = num_stages - 1
        total_ticks = warmup_ticks + steady_ticks + cooldown_ticks
    else:
        # GPipe
        total_ticks = 2 * (num_microbatches + num_stages - 1)

    useful_ticks = 2 * num_microbatches
    total_time = total_ticks * time_per_microbatch_sec
    efficiency = useful_ticks / total_ticks if total_ticks > 0 else 0

    return {
        "total_step_time_sec": round(total_time, 2),
        "efficiency": round(efficiency, 4),
        "bubble_time_fraction": round(1.0 - efficiency, 4),
        "total_ticks": total_ticks,
        "useful_ticks": useful_ticks,
    }


import math  # needed for compute_optimal_microbatches
