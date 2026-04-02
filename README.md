# Zeitgeist

Zeitgeist is a runtime for a backend-neutral distributed inference protocol.

It includes typed protocol objects, compatibility and planning logic, backend adapters, HTTP APIs, peer transports, and a deterministic test setup for validating the runtime without requiring real model servers.

## Features

- Typed protocol objects for nodes, backends, models, kernels, tensors, cache blobs, jobs, sessions, and events
- Compatibility reporting and execution planning with topology-aware cost ordering, trust-aware backend exclusion, and memory-pressure-aware repartitioning
- Backend adapters for `mlx` and `vllm`
- HTTP API for discovery, planning, execution, events, and runtime inspection
- Peer transport over TCP, QUIC, and Unix domain sockets
- Tensor and cache roundtrip framing with checksum validation
- Solo, tensor-parallel, expert-parallel, and hybrid distributed execution modes
- Per-attempt execution telemetry on job records
- Same-peer retry, active replan, failover, and fallback execution
- Deterministic synthetic mode for local testing

## CLI

```bash
cargo run -- serve --bind 127.0.0.1:8080
cargo run -- describe --pretty
cargo run -- smoke --prompt "hello mesh"
```

Peer commands:

```bash
cargo run -- serve-peer --bind 127.0.0.1:9090
cargo run -- serve-peer-quic --bind 127.0.0.1:9443 --cert /path/to/cert.pem --key /path/to/key.pem
cargo run -- serve-peer-unix --path /tmp/zeitgeist-peer.sock
cargo run -- peer-ping --addr 127.0.0.1:9090
cargo run -- peer-capabilities --addr 127.0.0.1:9090
cargo run -- peer-execute --addr 127.0.0.1:9090 --prompt "remote execute"
```

## API

- `GET /healthz`
- `GET /v1/version-policy`
- `GET /v1/node`
- `GET /v1/backends`
- `GET /v1/models`
- `GET /v1/kernels`
- `GET /v1/sessions`
- `GET /v1/transport-health`
- `GET /v1/planner-decisions`
- `GET /v1/topology`
- `GET /v1/schema`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/events`
- `GET /v1/events/stream`
- `POST /v1/compatibility`
- `POST /v1/plan`
- `POST /v1/jobs`
- `POST /v1/jobs/stream`
- `POST /v1/jobs/{job_id}/cancel`
- `POST /v1/tensors/roundtrip`
- `POST /v1/cache/roundtrip`

## Config

The default config is in [zeitgeist.toml](/Users/deepsaint/Desktop/zeitgeist/zeitgeist.toml). By default the runtime uses synthetic `mlx` and `vllm` backends and a shared dev token.

Protocol matching is strict:

- clients must send `x-zeitgeist-protocol-version`
- compatibility mode is `exact_only`
- backwards compatibility is disabled

## Validation

```bash
cargo test
fozzy doctor --deep --scenario tests/run.pass.fozzy.json --runs 5 --seed 1337 --json
fozzy test --det --strict tests/run.pass.fozzy.json tests/memory.pass.fozzy.json tests/distributed.pass.fozzy.json --json
fozzy run tests/run.pass.fozzy.json --det --record .fozzy/traces/run.fozzy --json
fozzy trace verify .fozzy/traces/run.fozzy --strict --json
fozzy replay .fozzy/traces/run.fozzy --json
fozzy ci .fozzy/traces/run.fozzy --json
```
