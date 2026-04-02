"""
P2P Discovery Service — WebRTC NAT Traversal for Miner Onboarding
====================================================================

Enables miners behind home routers (NAT) to discover and communicate with
their pipeline peers without manual port forwarding.

Architecture:
  1. Mining service acts as WebRTC signaling server
  2. Miners connect via WebSocket, exchange ICE candidates
  3. Once WebRTC DataChannel is established, miners use it for:
     - Weight shard transfers (replacing HTTP POST)
     - Pipeline control messages (activations, gradients)
     - Heartbeats and bandwidth probes

This makes onboarding "plug-and-play" - miners just need internet access.

Key components:
  - P2PDiscovery: manages signaling and peer matching
  - WebRTCConnection: wraps aiortc peer connection
  - DataChannelHandler: routes messages to existing services
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger("rg-mining.p2p-discovery")


@dataclass
class PeerInfo:
    """Information about a mining peer."""
    peer_id: str
    miner_id: str
    websocket: Optional[Any] = None  # WebSocket connection for signaling
    webrtc_connection: Optional[Any] = None  # aiortc RTCPeerConnection
    data_channel: Optional[Any] = None  # RTCDataChannel for messages
    ice_candidates: List[Dict] = field(default_factory=list)
    offer: Optional[Dict] = None
    answer: Optional[Dict] = None
    pipeline_group_id: Optional[str] = None
    stage_index: Optional[int] = None
    upstream_peer_id: Optional[str] = None
    downstream_peer_id: Optional[str] = None
    connected_at: Optional[str] = None


@dataclass
class SignalingMessage:
    """WebRTC signaling message."""
    type: str  # "offer", "answer", "ice-candidate", "peer-info"
    from_peer_id: str
    to_peer_id: Optional[str] = None
    data: Dict = field(default_factory=dict)


class P2PDiscovery:
    """
    WebRTC signaling server for miner P2P discovery.
    
    Manages the handshake process:
      1. Miner connects via WebSocket
      2. Server matches miner with pipeline peers
      3. Exchange WebRTC offers/answers/ICE candidates
      4. Once DataChannel opens, route messages to existing services
    """
    
    def __init__(self):
        self.peers: Dict[str, PeerInfo] = {}  # peer_id -> PeerInfo
        self.miner_to_peer: Dict[str, str] = {}  # miner_id -> peer_id
        self.pipeline_peers: Dict[str, Set[str]] = {}  # pipeline_group_id -> set of peer_ids
        self.pending_connections: Dict[Tuple[str, str], bool] = {}  # (peer1, peer2) -> attempting
        
    def register_miner(self, miner_id: str, websocket: Any) -> str:
        """Register a miner and assign a peer ID."""
        peer_id = f"peer-{uuid4().hex[:12]}"
        peer_info = PeerInfo(
            peer_id=peer_id,
            miner_id=miner_id,
            websocket=websocket,
        )
        self.peers[peer_id] = peer_info
        self.miner_to_peer[miner_id] = peer_id
        logger.info(f"Registered miner {miner_id} as peer {peer_id}")
        return peer_id
    
    def unregister_miner(self, miner_id: str) -> Optional[str]:
        """Unregister a miner and cleanup connections."""
        peer_id = self.miner_to_peer.pop(miner_id, None)
        if not peer_id:
            return None
        
        peer_info = self.peers.pop(peer_id, None)
        if peer_info:
            # Close WebRTC connection
            if peer_info.webrtc_connection:
                try:
                    peer_info.webrtc_connection.close()
                except Exception as e:
                    logger.warning(f"Failed to close WebRTC connection for {peer_id}: {e}")
            
            # Remove from pipeline groups
            if peer_info.pipeline_group_id:
                group_peers = self.pipeline_peers.get(peer_info.pipeline_group_id, set())
                group_peers.discard(peer_id)
                if not group_peers:
                    self.pipeline_peers.pop(peer_info.pipeline_group_id, None)
        
        logger.info(f"Unregistered miner {miner_id} (peer {peer_id})")
        return peer_id
    
    def assign_pipeline_peers(self, pipeline_group_id: str, assignments: List[Dict]):
        """
        Assign miners to pipeline groups and initiate P2P connections.
        
        Called after pipeline formation in ShardManager.
        """
        # Extract peer IDs for this pipeline
        group_peers = set()
        for assignment in assignments:
            miner_id = assignment["miner_id"]
            peer_id = self.miner_to_peer.get(miner_id)
            if peer_id:
                group_peers.add(peer_id)
                peer_info = self.peers.get(peer_id)
                if peer_info:
                    peer_info.pipeline_group_id = pipeline_group_id
                    peer_info.stage_index = assignment["stage_index"]
        
        self.pipeline_peers[pipeline_group_id] = group_peers
        
        # Establish upstream/downstream links
        sorted_peers = sorted(
            [(pid, self.peers[pid].stage_index) for pid in group_peers],
            key=lambda x: x[1]
        )
        
        for i, (peer_id, stage_idx) in enumerate(sorted_peers):
            peer_info = self.peers[peer_id]
            # Link to upstream peer
            if i > 0:
                peer_info.upstream_peer_id = sorted_peers[i-1][0]
            # Link to downstream peer  
            if i < len(sorted_peers) - 1:
                peer_info.downstream_peer_id = sorted_peers[i+1][0]
        
        # Initiate WebRTC connections between adjacent peers
        for i, (peer_id, stage_idx) in enumerate(sorted_peers):
            if i < len(sorted_peers) - 1:
                downstream_peer_id = sorted_peers[i+1][0]
                self._initiate_webrtc_connection(peer_id, downstream_peer_id)
        
        logger.info(f"Assigned {len(group_peers)} peers to pipeline {pipeline_group_id}")
    
    def _initiate_webrtc_connection(self, initiator_peer_id: str, target_peer_id: str):
        """Start WebRTC connection between two peers."""
        if (initiator_peer_id, target_peer_id) in self.pending_connections:
            return  # Already attempting
        
        self.pending_connections[(initiator_peer_id, target_peer_id)] = True
        
        # Tell initiator to create offer
        initiator_peer = self.peers.get(initiator_peer_id)
        target_peer = self.peers.get(target_peer_id)
        
        if not initiator_peer or not target_peer:
            logger.warning(f"Cannot initiate WebRTC: missing peer info")
            return
        
        logger.info(f"Initiating WebRTC: {initiator_peer_id} -> {target_peer_id}")
        
        # Send offer request to initiator
        message = SignalingMessage(
            type="create-offer",
            from_peer_id=initiator_peer_id,
            to_peer_id=target_peer_id,
            data={
                "target_peer_id": target_peer_id,
                "target_miner_id": target_peer.miner_id,
            }
        )
        self._send_signaling_message(initiator_peer.websocket, message)
    
    def handle_signaling_message(self, miner_id: str, message: Dict):
        """Handle incoming WebRTC signaling message."""
        peer_id = self.miner_to_peer.get(miner_id)
        if not peer_id:
            logger.warning(f"Signaling from unregistered miner {miner_id}")
            return
        
        msg_type = message.get("type")
        
        # Support both flat and nested message formats
        if "data" in message:
            data = message["data"]
        else:
            # Flat format - use the whole message except type
            data = {k: v for k, v in message.items() if k != "type"}
        
        if msg_type == "offer":
            self._handle_offer(peer_id, data)
        elif msg_type == "answer":
            self._handle_answer(peer_id, data)
        elif msg_type == "ice-candidate":
            self._handle_ice_candidate(peer_id, data)
        elif msg_type == "datachannel-open":
            self._handle_datachannel_open(peer_id, data)
        else:
            logger.warning(f"Unknown signaling message type: {msg_type}")
    
    def _handle_offer(self, from_peer_id: str, data: Dict):
        """Forward WebRTC offer to target peer."""
        to_peer_id = data.get("target_peer_id")
        if not to_peer_id:
            logger.warning("Offer missing target_peer_id")
            return
        
        to_peer = self.peers.get(to_peer_id)
        if not to_peer or not to_peer.websocket:
            logger.warning(f"Target peer {to_peer_id} not available")
            return
        
        # Store offer
        from_peer = self.peers.get(from_peer_id)
        if from_peer:
            from_peer.offer = data
        
        # Forward to target
        message = SignalingMessage(
            type="offer",
            from_peer_id=from_peer_id,
            to_peer_id=to_peer_id,
            data=data
        )
        self._send_signaling_message(to_peer.websocket, message)
        logger.info(f"Forwarded offer from {from_peer_id} to {to_peer_id}")
    
    def _handle_answer(self, from_peer_id: str, data: Dict):
        """Forward WebRTC answer to target peer."""
        to_peer_id = data.get("target_peer_id")
        if not to_peer_id:
            logger.warning("Answer missing target_peer_id")
            return
        
        to_peer = self.peers.get(to_peer_id)
        if not to_peer or not to_peer.websocket:
            logger.warning(f"Target peer {to_peer_id} not available")
            return
        
        # Store answer
        from_peer = self.peers.get(from_peer_id)
        if from_peer:
            from_peer.answer = data
        
        # Forward to target
        message = SignalingMessage(
            type="answer",
            from_peer_id=from_peer_id,
            to_peer_id=to_peer_id,
            data=data
        )
        self._send_signaling_message(to_peer.websocket, message)
        logger.info(f"Forwarded answer from {from_peer_id} to {to_peer_id}")
    
    def _handle_ice_candidate(self, from_peer_id: str, data: Dict):
        """Forward ICE candidate to target peer."""
        to_peer_id = data.get("target_peer_id")
        if not to_peer_id:
            logger.warning("ICE candidate missing target_peer_id")
            return
        
        to_peer = self.peers.get(to_peer_id)
        if not to_peer or not to_peer.websocket:
            logger.warning(f"Target peer {to_peer_id} not available")
            return
        
        # Store candidate
        from_peer = self.peers.get(from_peer_id)
        if from_peer:
            from_peer.ice_candidates.append(data)
        
        # Forward to target
        message = SignalingMessage(
            type="ice-candidate",
            from_peer_id=from_peer_id,
            to_peer_id=to_peer_id,
            data=data
        )
        self._send_signaling_message(to_peer.websocket, message)
    
    def _handle_datachannel_open(self, peer_id: str, data: Dict):
        """DataChannel opened - mark peer as connected."""
        peer_info = self.peers.get(peer_id)
        if peer_info:
            peer_info.connected_at = data.get("connected_at")
            logger.info(f"DataChannel open for peer {peer_id} (miner {peer_info.miner_id})")
    
    def _send_signaling_message(self, websocket: Any, message: SignalingMessage):
        """Send signaling message via WebSocket."""
        try:
            msg_dict = {
                "type": message.type,
                "from_peer_id": message.from_peer_id,
                "to_peer_id": message.to_peer_id,
                "data": message.data,
            }
            # Handle both sync and async send_text
            if asyncio.iscoroutinefunction(websocket.send_text):
                asyncio.create_task(websocket.send_text(json.dumps(msg_dict)))
            else:
                websocket.send_text(json.dumps(msg_dict))
        except Exception as e:
            logger.error(f"Failed to send signaling message: {e}")
    
    def get_peer_info(self, miner_id: str) -> Optional[Dict]:
        """Get peer information for a miner."""
        peer_id = self.miner_to_peer.get(miner_id)
        if not peer_id:
            return None
        
        peer_info = self.peers.get(peer_id)
        if not peer_info:
            return None
        
        return {
            "peer_id": peer_info.peer_id,
            "miner_id": peer_info.miner_id,
            "pipeline_group_id": peer_info.pipeline_group_id,
            "stage_index": peer_info.stage_index,
            "upstream_peer_id": peer_info.upstream_peer_id,
            "downstream_peer_id": peer_info.downstream_peer_id,
            "connected_at": peer_info.connected_at,
            "has_datachannel": peer_info.data_channel is not None,
        }
    
    def get_pipeline_status(self, pipeline_group_id: str) -> Dict:
        """Get connection status for all peers in a pipeline."""
        peer_ids = self.pipeline_peers.get(pipeline_group_id, set())
        peers = []
        connected_count = 0
        
        for peer_id in peer_ids:
            peer_info = self.peers.get(peer_id)
            if peer_info:
                peers.append({
                    "peer_id": peer_id,
                    "miner_id": peer_info.miner_id,
                    "stage_index": peer_info.stage_index,
                    "connected": peer_info.connected_at is not None,
                    "has_datachannel": peer_info.data_channel is not None,
                })
                if peer_info.connected_at:
                    connected_count += 1
        
        return {
            "pipeline_group_id": pipeline_group_id,
            "total_peers": len(peers),
            "connected_peers": connected_count,
            "peers": peers,
        }


# Global instance
p2p_discovery = P2PDiscovery()
