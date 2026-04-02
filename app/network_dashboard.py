"""
Network Dashboard for Resonant Genesis Mining
============================================

Real-time monitoring dashboard for:
- WebRTC P2P mesh topology and health
- Miner reputation and slashing statistics
- Network bandwidth and transfer metrics
- Wallet balances and staking information
- Live violation tracking and penalties

Provides WebSocket streaming for real-time updates and REST API
for historical data and detailed metrics.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .p2p_discovery import p2p_discovery
from .slashing import slashing_engine
from .weight_shard_registry import weight_registry
from .wallet_service import wallet_service


logger = logging.getLogger("rg-mining.dashboard")


@dataclass
class NetworkNode:
    """Represents a node in the network topology."""
    miner_id: str
    peer_id: str
    status: str  # connected, disconnected, suspended
    bandwidth_mbps: float
    connected_peers: int
    webrtc_enabled: bool
    reputation_score: int
    last_seen: str
    location: Dict[str, float] = None  # lat, lng for visualization


@dataclass
class NetworkLink:
    """Represents a connection between two nodes."""
    source: str
    target: str
    bandwidth_mbps: float
    latency_ms: float
    transfer_method: str  # webrtc, http
    active_transfers: int


@dataclass
class WalletInfo:
    """Wallet and staking information for a miner."""
    miner_id: str
    wallet_address: str
    stake_amount: float
    pending_slashes: float
    collateral_required: float
    reputation_score: int
    earnings_24h: float
    total_earnings: float


@dataclass
class NetworkMetrics:
    """Overall network health metrics."""
    total_miners: int
    active_miners: int
    suspended_miners: int
    total_bandwidth_gbps: float
    p2p_transfers_24h: int
    violations_24h: int
    avg_reputation: float
    network_health_score: float


class NetworkDashboard:
    """
    Real-time network dashboard with WebSocket streaming.
    
    Aggregates data from P2P discovery, slashing engine, and weight registry
    to provide comprehensive network monitoring.
    """
    
    def __init__(self):
        self.connected_clients: Dict[str, WebSocket] = {}
        self.last_update = time.time()
        self.update_interval = 5.0  # Update every 5 seconds
        
    async def register_client(self, websocket: WebSocket, client_id: str):
        """Register a WebSocket client for real-time updates."""
        await websocket.accept()
        self.connected_clients[client_id] = websocket
        logger.info(f"Dashboard client connected: {client_id}")
        
        # Send initial data
        await self.send_update(client_id)
    
    async def unregister_client(self, client_id: str):
        """Unregister a WebSocket client."""
        if client_id in self.connected_clients:
            del self.connected_clients[client_id]
            logger.info(f"Dashboard client disconnected: {client_id}")
    
    async def broadcast_update(self):
        """Broadcast network updates to all connected clients."""
        if not self.connected_clients:
            return
        
        # Check if it's time to update
        now = time.time()
        if now - self.last_update < self.update_interval:
            return
        
        self.last_update = now
        
        # Send updates to all clients
        for client_id in list(self.connected_clients.keys()):
            try:
                await self.send_update(client_id)
            except Exception as e:
                logger.error(f"Failed to send update to {client_id}: {e}")
                await self.unregister_client(client_id)
    
    async def send_update(self, client_id: str):
        """Send network update to a specific client."""
        if client_id not in self.connected_clients:
            return
        
        try:
            data = await self.get_dashboard_data()
            await self.connected_clients[client_id].send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Failed to send dashboard update: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nodes": await self._get_network_nodes(),
            "links": await self._get_network_links(),
            "metrics": await self._get_network_metrics(),
            "wallets": await self._get_wallet_info(),
            "violations": await self._get_recent_violations(),
            "topology": await self._get_network_topology(),
        }
    
    async def _get_network_nodes(self) -> List[Dict]:
        """Get all network nodes with their status."""
        nodes = []
        
        # Get P2P discovery info
        all_peers = p2p_discovery.get_all_peers()
        
        # Get slashing reputation info
        network_stats = slashing_engine.get_network_stats()
        
        for peer_id, peer_info in all_peers.items():
            miner_id = peer_info.get("miner_id", peer_id)
            
            # Get reputation from slashing engine
            reputation = slashing_engine.reputation_tracker.get(miner_id)
            if not reputation:
                reputation = type('obj', (object,), {
                    'score': 100,
                    'is_suspended_now': lambda: False,
                    'suspension_end': ''
                })()
            
            node = {
                "miner_id": miner_id,
                "peer_id": peer_id,
                "status": "suspended" if reputation.is_suspended_now() else 
                         "connected" if peer_info.get("connected") else "disconnected",
                "bandwidth_mbps": peer_info.get("bandwidth_mbps", 0),
                "connected_peers": len(peer_info.get("connected_peers", [])),
                "webrtc_enabled": peer_info.get("has_datachannel", False),
                "reputation_score": reputation.score,
                "last_seen": peer_info.get("connected_at", ""),
                "location": self._get_miner_location(miner_id),
            }
            nodes.append(node)
        
        return nodes
    
    async def _get_network_links(self) -> List[Dict]:
        """Get all network connections/links."""
        links = []
        
        # Get P2P connections
        all_peers = p2p_discovery.get_all_peers()
        
        for peer_id, peer_info in all_peers.items():
            connected_peers = peer_info.get("connected_peers", [])
            bandwidth = peer_info.get("bandwidth_mbps", 0)
            
            for connected_peer in connected_peers:
                # Create bidirectional link
                link = {
                    "source": peer_id,
                    "target": connected_peer,
                    "bandwidth_mbps": bandwidth,
                    "latency_ms": peer_info.get("latency_ms", 50),
                    "transfer_method": "webrtc" if peer_info.get("has_datachannel") else "http",
                    "active_transfers": peer_info.get("active_transfers", 0),
                }
                links.append(link)
        
        return links
    
    async def _get_network_metrics(self) -> Dict:
        """Get overall network health metrics."""
        # Get stats from all services
        p2p_stats = p2p_discovery.get_network_stats()
        slashing_stats = slashing_engine.get_network_stats()
        weight_stats = weight_registry.get_registry_stats()
        
        total_miners = slashing_stats["total_miners"]
        active_miners = len([p for p in p2p_discovery.get_all_peers().values() if p.get("connected")])
        suspended_miners = slashing_stats["suspended_miners"]
        
        # Calculate total bandwidth
        total_bandwidth = sum(p.get("bandwidth_mbps", 0) for p in p2p_discovery.get_all_peers().values())
        
        # Calculate network health score (0-100)
        health_score = (
            (active_miners / max(total_miners, 1)) * 40 +  # Active miners (40%)
            (slashing_stats["avg_reputation"] / 100) * 30 +  # Reputation (30%)
            (min(total_bandwidth / 1000, 1) * 30)  # Bandwidth (30%)
        )
        
        return {
            "total_miners": total_miners,
            "active_miners": active_miners,
            "suspended_miners": suspended_miners,
            "total_bandwidth_gbps": total_bandwidth / 1000,
            "p2p_transfers_24h": weight_stats.get("total_p2p_transfers", 0),
            "violations_24h": len([v for v in slashing_engine.violations 
                                 if datetime.fromisoformat(v.reported_at.replace('Z', '+00:00')) > 
                                 datetime.now(timezone.utc) - timedelta(hours=24)]),
            "avg_reputation": slashing_stats["avg_reputation"],
            "network_health_score": round(health_score, 1),
        }
    
    async def _get_wallet_info(self) -> List[Dict]:
        """Get wallet and staking information for all miners."""
        wallets = []
        
        # Get all miners from reputation tracker
        for miner_id, reputation in slashing_engine.reputation_tracker.items():
            # Get wallet from wallet service
            wallet = await wallet_service.get_wallet(miner_id)
            if wallet:
                stats = await wallet_service.get_wallet_stats(wallet.wallet_address)
                
                # Get RG token balance
                rg_balance = await wallet_service.get_balance(wallet.wallet_address, wallet_service.TokenType.RG_TOKEN)
                
                wallet_info = {
                    "miner_id": miner_id,
                    "wallet_address": wallet.wallet_address,
                    "stake_amount": sum(s.amount for s in stats["stakes"] if s["status"] == "active"),
                    "pending_slashes": sum(v.slash_amount for v in slashing_engine.violations 
                                          if v.miner_id == miner_id and not v.slashed),
                    "collateral_required": reputation.collateral_required,
                    "reputation_score": reputation.score,
                    "earnings_24h": stats["rewards_24h"],
                    "total_earnings": sum(b["balance"] for b in stats["balances"] if b["token_type"] == "rg_token"),
                    "total_balance_usd": stats["total_balance_usd"],
                }
            else:
                # Create mock wallet if not exists
                wallet_info = {
                    "miner_id": miner_id,
                    "wallet_address": f"0x{miner_id[:8]}...{miner_id[-8:]}",
                    "stake_amount": reputation.collateral_required,
                    "pending_slashes": sum(v.slash_amount for v in slashing_engine.violations 
                                          if v.miner_id == miner_id and not v.slashed),
                    "collateral_required": reputation.collateral_required,
                    "reputation_score": reputation.score,
                    "earnings_24h": reputation.score * 0.1,  # Mock earnings
                    "total_earnings": reputation.score * 10.0,  # Mock total
                    "total_balance_usd": reputation.collateral_required,
                }
            
            wallets.append(wallet_info)
        
        return wallets
    
    async def _get_recent_violations(self) -> List[Dict]:
        """Get recent violations for display."""
        violations = sorted(
            slashing_engine.violations,
            key=lambda v: v.reported_at,
            reverse=True
        )[:20]  # Last 20 violations
        
        return [
            {
                "record_id": v.record_id,
                "miner_id": v.miner_id,
                "violation_type": v.violation_type.value,
                "severity": v.severity.value,
                "reported_at": v.reported_at,
                "verified": v.verified,
                "slashed": v.slashed,
                "slash_amount": v.slash_amount,
                "suspension_end": v.suspension_end,
            }
            for v in violations
        ]
    
    async def _get_network_topology(self) -> Dict:
        """Get network topology for visualization."""
        nodes = await self._get_network_nodes()
        links = await self._get_network_links()
        
        # Calculate layout positions (simple force-directed layout)
        positions = {}
        for i, node in enumerate(nodes):
            angle = (2 * 3.14159 * i) / len(nodes)
            radius = 300
            positions[node["peer_id"]] = {
                "x": radius * (1 + 0.3 * (node["reputation_score"] / 100)) * (angle / 3.14159),
                "y": radius * (1 + 0.3 * (node["reputation_score"] / 100)) * (angle / 3.14159),
            }
        
        return {
            "nodes": nodes,
            "links": links,
            "positions": positions,
        }
    
    def _get_miner_location(self, miner_id: str) -> Dict[str, float]:
        """Get mock location for miner visualization."""
        # Simple hash-based location for demo
        hash_val = hash(miner_id) % 360
        return {
            "lat": 40.7128 + (hash_val % 20 - 10) * 0.1,  # Around NYC
            "lng": -74.0060 + (hash_val % 20 - 10) * 0.1,
        }


# Global dashboard instance
network_dashboard = NetworkDashboard()


# HTML template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resonant Genesis Network Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            overflow-x: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 20px;
            border-bottom: 2px solid #00ff88;
        }
        
        .header h1 {
            font-size: 2.5em;
            background: linear-gradient(45deg, #00ff88, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #888;
            font-size: 1.1em;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .card {
            background: #1a1a2e;
            border-radius: 10px;
            padding: 20px;
            border: 1px solid #333;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        
        .card h2 {
            color: #00ff88;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background: #0f0f1e;
            border-radius: 5px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #00ccff;
        }
        
        .network-topology {
            grid-column: span 2;
            height: 400px;
            position: relative;
        }
        
        .violations {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .violation-item {
            padding: 10px;
            margin: 5px 0;
            background: #0f0f1e;
            border-radius: 5px;
            border-left: 4px solid #ff4444;
        }
        
        .violation-item.verified {
            border-left-color: #ff8844;
        }
        
        .violation-item.slashed {
            border-left-color: #00ff88;
        }
        
        .wallet-info {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .wallet-item {
            padding: 10px;
            margin: 5px 0;
            background: #0f0f1e;
            border-radius: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-connected { background: #00ff88; }
        .status-disconnected { background: #ff4444; }
        .status-suspended { background: #ff8844; }
        
        .health-score {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        
        .health-score.high { color: #00ff88; }
        .health-score.medium { color: #ffaa00; }
        .health-score.low { color: #ff4444; }
        
        .topology-svg {
            width: 100%;
            height: 100%;
        }
        
        .node {
            cursor: pointer;
        }
        
        .node circle {
            stroke: #fff;
            stroke-width: 2px;
        }
        
        .node text {
            fill: #fff;
            font-size: 12px;
            text-anchor: middle;
        }
        
        .link {
            stroke: #666;
            stroke-width: 2px;
        }
        
        .link.webrtc {
            stroke: #00ff88;
        }
        
        .link.http {
            stroke: #ff8844;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .updating {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Resonant Genesis Network Dashboard</h1>
        <p>Real-time monitoring of P2P weight distribution network</p>
    </div>
    
    <div class="dashboard">
        <!-- Network Health Card -->
        <div class="card">
            <h2>Network Health</h2>
            <div class="health-score" id="healthScore">--</div>
            <div class="metric">
                <span>Total Miners</span>
                <span class="metric-value" id="totalMiners">--</span>
            </div>
            <div class="metric">
                <span>Active Miners</span>
                <span class="metric-value" id="activeMiners">--</span>
            </div>
            <div class="metric">
                <span>Suspended Miners</span>
                <span class="metric-value" id="suspendedMiners">--</span>
            </div>
            <div class="metric">
                <span>Total Bandwidth</span>
                <span class="metric-value" id="totalBandwidth">--</span>
            </div>
            <div class="metric">
                <span>Avg Reputation</span>
                <span class="metric-value" id="avgReputation">--</span>
            </div>
        </div>
        
        <!-- Network Topology -->
        <div class="card network-topology">
            <h2>Network Topology</h2>
            <svg class="topology-svg" id="topologySvg"></svg>
        </div>
        
        <!-- Recent Violations -->
        <div class="card violations">
            <h2>Recent Violations</h2>
            <div id="violationsList"></div>
        </div>
        
        <!-- Wallet Information -->
        <div class="card wallet-info">
            <h2>Miner Wallets</h2>
            <div id="walletsList"></div>
        </div>
        
        <!-- Transfer Metrics -->
        <div class="card">
            <h2>Transfer Metrics</h2>
            <div class="metric">
                <span>P2P Transfers (24h)</span>
                <span class="metric-value" id="p2pTransfers">--</span>
            </div>
            <div class="metric">
                <span>Violations (24h)</span>
                <span class="metric-value" id="violations24h">--</span>
            </div>
            <div class="metric">
                <span>WebRTC Links</span>
                <span class="metric-value" id="webrtcLinks">--</span>
            </div>
            <div class="metric">
                <span>HTTP Links</span>
                <span class="metric-value" id="httpLinks">--</span>
            </div>
            <div class="metric">
                <span>Active Transfers</span>
                <span class="metric-value" id="activeTransfers">--</span>
            </div>
        </div>
    </div>
    
    <script>
        class NetworkDashboard {
            constructor() {
                this.ws = null;
                this.data = null;
                this.init();
            }
            
            init() {
                this.connectWebSocket();
                this.setupTopology();
            }
            
            connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/mining/dashboard/ws`;
                
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    console.log('Dashboard WebSocket connected');
                    document.body.classList.add('updating');
                };
                
                this.ws.onmessage = (event) => {
                    this.data = JSON.parse(event.data);
                    this.updateDashboard();
                    document.body.classList.remove('updating');
                };
                
                this.ws.onclose = () => {
                    console.log('Dashboard WebSocket disconnected');
                    setTimeout(() => this.connectWebSocket(), 5000);
                };
                
                this.ws.onerror = (error) => {
                    console.error('Dashboard WebSocket error:', error);
                };
            }
            
            updateDashboard() {
                if (!this.data) return;
                
                // Update network health
                const healthScore = this.data.metrics.network_health_score;
                const healthElement = document.getElementById('healthScore');
                healthElement.textContent = healthScore.toFixed(1);
                healthElement.className = 'health-score ' + 
                    (healthScore >= 80 ? 'high' : healthScore >= 50 ? 'medium' : 'low');
                
                // Update metrics
                document.getElementById('totalMiners').textContent = this.data.metrics.total_miners;
                document.getElementById('activeMiners').textContent = this.data.metrics.active_miners;
                document.getElementById('suspendedMiners').textContent = this.data.metrics.suspended_miners;
                document.getElementById('totalBandwidth').textContent = 
                    this.data.metrics.total_bandwidth_gbps.toFixed(2) + ' Gbps';
                document.getElementById('avgReputation').textContent = 
                    this.data.metrics.avg_reputation.toFixed(1);
                
                // Update transfer metrics
                document.getElementById('p2pTransfers').textContent = this.data.metrics.p2p_transfers_24h;
                document.getElementById('violations24h').textContent = this.data.metrics.violations_24h;
                
                const webrtcLinks = this.data.links.filter(l => l.transfer_method === 'webrtc').length;
                const httpLinks = this.data.links.filter(l => l.transfer_method === 'http').length;
                document.getElementById('webrtcLinks').textContent = webrtcLinks;
                document.getElementById('httpLinks').textContent = httpLinks;
                
                const activeTransfers = this.data.links.reduce((sum, l) => sum + l.active_transfers, 0);
                document.getElementById('activeTransfers').textContent = activeTransfers;
                
                // Update violations
                this.updateViolations();
                
                // Update wallets
                this.updateWallets();
                
                // Update topology
                this.updateTopology();
            }
            
            updateViolations() {
                const violationsList = document.getElementById('violationsList');
                violationsList.innerHTML = '';
                
                this.data.violations.forEach(violation => {
                    const item = document.createElement('div');
                    item.className = 'violation-item';
                    if (violation.verified) item.classList.add('verified');
                    if (violation.slashed) item.classList.add('slashed');
                    
                    item.innerHTML = `
                        <div style="display: flex; justify-content: space-between;">
                            <span>${violation.miner_id}</span>
                            <span style="color: #ff8844;">${violation.violation_type}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #888;">
                            ${new Date(violation.reported_at).toLocaleString()}
                            ${violation.slashed ? `• Slashed: ${violation.slash_amount}%` : ''}
                        </div>
                    `;
                    
                    violationsList.appendChild(item);
                });
            }
            
            updateWallets() {
                const walletsList = document.getElementById('walletsList');
                walletsList.innerHTML = '';
                
                this.data.wallets.forEach(wallet => {
                    const item = document.createElement('div');
                    item.className = 'wallet-item';
                    
                    const statusClass = wallet.reputation_score >= 80 ? 'status-connected' :
                                      wallet.reputation_score >= 50 ? 'status-disconnected' : 'status-suspended';
                    
                    item.innerHTML = `
                        <div style="display: flex; justify-content: space-between;">
                            <span>
                                <span class="status-indicator ${statusClass}"></span>
                                ${wallet.miner_id}
                            </span>
                            <span style="color: #00ccff;">${wallet.wallet_address}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #888; margin-top: 5px;">
                            Stake: ${wallet.stake_amount} • 
                            Rep: ${wallet.reputation_score} • 
                            Pending: ${wallet.pending_slashes}
                        </div>
                    `;
                    
                    walletsList.appendChild(item);
                });
            }
            
            setupTopology() {
                const svg = d3.select('#topologySvg');
                const width = svg.node().getBoundingClientRect().width;
                const height = svg.node().getBoundingClientRect().height;
                
                this.topologySimulation = d3.forceSimulation()
                    .force('link', d3.forceLink().id(d => d.peer_id).distance(100))
                    .force('charge', d3.forceManyBody().strength(-300))
                    .force('center', d3.forceCenter(width / 2, height / 2))
                    .force('collision', d3.forceCollide().radius(30));
            }
            
            updateTopology() {
                const svg = d3.select('#topologySvg');
                const width = svg.node().getBoundingClientRect().width;
                const height = svg.node().getBoundingClientRect().height;
                
                // Clear existing
                svg.selectAll('*').remove();
                
                // Create links
                const link = svg.append('g')
                    .selectAll('line')
                    .data(this.data.links)
                    .enter().append('line')
                    .attr('class', d => `link ${d.transfer_method}`)
                    .attr('stroke-width', d => Math.max(1, d.bandwidth_mbps / 100));
                
                // Create nodes
                const node = svg.append('g')
                    .selectAll('g')
                    .data(this.data.nodes)
                    .enter().append('g')
                    .attr('class', 'node');
                
                node.append('circle')
                    .attr('r', d => 10 + d.reputation_score / 10)
                    .attr('fill', d => {
                        if (d.status === 'suspended') return '#ff8844';
                        if (d.status === 'disconnected') return '#ff4444';
                        return d.webrtc_enabled ? '#00ff88' : '#00ccff';
                    });
                
                node.append('text')
                    .text(d => d.miner_id.substring(0, 8))
                    .attr('dy', -15);
                
                // Update simulation
                this.topologySimulation.nodes(this.data.nodes);
                this.topologySimulation.force('link').links(this.data.links);
                
                this.topologySimulation.on('tick', () => {
                    link
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    node
                        .attr('transform', d => `translate(${d.x},${d.y})`);
                });
                
                this.topologySimulation.alpha(1).restart();
            }
        }
        
        // Initialize dashboard
        const dashboard = new NetworkDashboard();
    </script>
</body>
</html>
"""


# Global dashboard instance for router
network_dashboard = NetworkDashboard()
