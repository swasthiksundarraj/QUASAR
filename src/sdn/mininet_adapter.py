from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import platform
import re
import subprocess
from typing import Any

from sdn.telemetry import FlowTelemetry


_COUNTER = re.compile(r"\b(n_packets|n_bytes)=([0-9]+)\b")
_DURATION = re.compile(r"\bduration=([0-9.]+)s\b")
_IN_PORT = re.compile(r"\bin_port=([0-9]+)\b")
_IP_PROTO = re.compile(r"\b(?:ip_proto|nw_proto)=([0-9]+)\b")


class MininetStatsAdapter:
    """Optional Linux-only OVS statistics collector; it generates no traffic."""

    def __init__(self, controller_ip: str = "127.0.0.1", controller_port: int = 6653) -> None:
        import ipaddress
        ipaddress.ip_address(controller_ip)
        if not 1 <= controller_port <= 65535:
            raise ValueError("controller_port must be in [1, 65535]")
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.network: Any | None = None

    def start(self) -> None:
        if platform.system() != "Linux":
            raise RuntimeError("Mininet/Open vSwitch emulation requires an isolated Linux VM")
        import os
        if os.geteuid() != 0:
            raise PermissionError("Mininet startup requires root inside the isolated Linux VM")
        try:
            from mininet.topo import SingleSwitchTopo
            from mininet.net import Mininet
            from mininet.node import OVSSwitch, RemoteController
        except ImportError as exc:
            raise RuntimeError("Install Mininet and Open vSwitch in the isolated Linux VM") from exc
        network = Mininet(
            topo=SingleSwitchTopo(n=2),
            controller=RemoteController,
            switch=OVSSwitch,
            autoSetMacs=True,
        )
        network.addController("c0", controller=RemoteController, ip=self.controller_ip, port=self.controller_port)
        network.get("s1").protocols = "OpenFlow13"
        network.get("s1").failMode = "secure"
        network.build()
        network.start()
        self.network = network

    def collect(self) -> list[FlowTelemetry]:
        if self.network is None:
            raise RuntimeError("Start the Mininet lab before collecting flow statistics")
        result = subprocess.run(
            ["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", "s1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        now = datetime.now(timezone.utc).isoformat()
        records = []
        for line in result.stdout.splitlines():
            counters = {name: int(value) for name, value in _COUNTER.findall(line)}
            if "n_packets" not in counters or "n_bytes" not in counters:
                continue
            duration_match = _DURATION.search(line)
            port_match = _IN_PORT.search(line)
            protocol_match = _IP_PROTO.search(line)
            protocol = {
                1: "ICMP", 6: "TCP", 17: "UDP", 58: "ICMPv6",
            }.get(int(protocol_match.group(1)), "OTHER") if protocol_match else "OTHER"
            flow_id = hashlib.sha256(line.strip().encode()).hexdigest()[:16]
            records.append(FlowTelemetry(
                timestamp=now,
                switch_id="s1",
                host_id=f"port-{port_match.group(1)}" if port_match else "unknown",
                flow_id=flow_id,
                packet_count=counters["n_packets"],
                byte_count=counters["n_bytes"],
                duration_seconds=float(duration_match.group(1)) if duration_match else 0.0,
                protocol=protocol,
                controller_event="ovs-flow-stats",
                synthetic=False,
            ))
        return records

    def close(self) -> None:
        if self.network is not None:
            self.network.stop()
            self.network = None
