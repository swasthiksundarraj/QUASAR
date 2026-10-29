# Optional isolated Mininet lab

The project host may be macOS; Mininet and Open vSwitch should be run in a disposable Linux VM with a host-only network, not on a production network. Install Mininet, Open vSwitch, and an OpenFlow 1.3-compatible controller inside that VM using the Linux distribution's documented packages. Verify the controller and switch protocol version compatibility there before connecting components.

The project supplies a tested-in-process mock adapter and an optional `MininetStatsAdapter`. The latter starts a two-host, one-switch OpenFlow 1.3 topology and reads `ovs-ofctl dump-flows` counters; it generates no traffic and does not apply live mitigation. It requires root inside Linux and a controller already listening at the configured address. The adapter has not been executed on this macOS host. Before connecting any lab adapter:

1. Keep the VM network isolated and use only synthetic/authorized test hosts.
2. Read flow counters from the selected controller API and normalize them to `sdn.telemetry.FlowTelemetry`.
3. Keep mitigation action types and switch/host scopes allowlisted; call `quasar.qubo.verify_plan` immediately before any controller operation.
4. Persist the plan, verifier decision, controller response, and rollback result to the SQLite audit log.
5. Test rollback and failure behavior with mock adapters before connecting a controller.

`sdn.telemetry.start_mininet_lab()` raises an explicit platform/setup error rather than pretending Mininet integration is present.
