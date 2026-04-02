# Zeitgeist

Zeitgeist is a runtime for a backend-neutral distributed inference protocol.

It includes typed protocol objects, compatibility and planning logic, backend adapters, HTTP APIs, peer transports, and deterministic certification paths for validating the runtime alongside live model execution.

## Features

- Typed protocol objects for nodes, backends, models, kernels, tensors, cache blobs, jobs, sessions, and events
- Compatibility reporting and execution planning with topology-aware cost ordering, trust-aware backend exclusion, and memory-pressure-aware repartitioning
- Live backend adapters for `mlx` and `vllm`
- HTTP API for discovery, planning, execution, events, and runtime inspection
- Peer transport over TCP, QUIC, and Unix domain sockets
- Signed peer identity on handshake and certificate-validated QUIC mTLS when configured
- Tensor and cache roundtrip framing with checksum validation
- Solo, tensor-parallel, expert-parallel, and hybrid distributed execution modes
- Per-attempt execution telemetry on job records
- Same-peer retry, active replan, failover, and fallback execution
- Deterministic synthetic mode as an explicit opt-in for testing and certification

## CLI

```bash
cargo run -- serve --bind 127.0.0.1:8080
cargo run -- describe --pretty
cargo run -- smoke --prompt "hello mesh"
```

Peer commands:

```bash
cargo run -- serve-peer --bind 127.0.0.1:9090
cargo run -- serve-peer-quic --bind 127.0.0.1:9443 --cert /path/to/cert.pem --key /path/to/key.pem --client-ca-cert /path/to/ca.pem
cargo run -- serve-peer-unix --path /tmp/zeitgeist-peer.sock
cargo run -- peer-ping --addr 127.0.0.1:9090
cargo run -- peer-ping-quic --addr 127.0.0.1:9443 --server-name localhost --ca-cert /path/to/ca.pem --client-cert /path/to/client-cert.pem --client-key /path/to/client-key.pem
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

The default config is in [zeitgeist.toml](/Users/deepsaint/Desktop/zeitgeist/zeitgeist.toml). By default the runtime uses live `mlx` and live `vllm` backends plus a shared dev token.

If no config file is supplied, the runtime also defaults to live backends and reads:

- `ZEITGEIST_MLX_PYTHON` or `python3`
- `ZEITGEIST_MLX_MODEL` or `mlx-community/Llama-3.2-1B-Instruct-4bit`
- `ZEITGEIST_VLLM_BASE_URL` or `http://127.0.0.1:8000`
- `ZEITGEIST_VLLM_API_KEY` when the upstream OpenAI-compatible server requires auth

Synthetic execution remains available only through explicit `kind = "synthetic"` backend entries in the runtime config.

The checked-in `vllm` backend assumes a live upstream server is reachable at `http://127.0.0.1:8000`.

Protocol matching is strict:

- clients must send `x-zeitgeist-protocol-version`
- compatibility mode is `exact_only`
- backwards compatibility is disabled

Peer trust is also strict:

- peer handshakes require a signed node identity
- strict planning excludes unsigned backend attestations
- QUIC can enforce mutual TLS with a configured client CA

## Validation

```bash
cargo test
fozzy doctor --deep --scenario tests/run.pass.fozzy.json --runs 5 --seed 1337 --json
fozzy test --det --strict tests/run.pass.fozzy.json tests/memory.pass.fozzy.json tests/distributed.pass.fozzy.json --json
fozzy run tests/run.pass.fozzy.json --det --record .fozzy/traces/run.fozzy --json
fozzy trace verify .fozzy/traces/run.fozzy --strict --json
fozzy replay .fozzy/traces/run.fozzy --json
fozzy ci .fozzy/traces/run.fozzy --json
fozzy doctor --deep --scenario tests/quic_mtls.pass.fozzy.json --runs 5 --seed 1337 --json
fozzy test --det --strict tests/quic_mtls.pass.fozzy.json tests/quic.pass.fozzy.json --json
fozzy run tests/quic_mtls.pass.fozzy.json --det --record .fozzy/traces/quic-mtls.fozzy --json
fozzy trace verify .fozzy/traces/quic-mtls.fozzy --strict --json
fozzy replay .fozzy/traces/quic-mtls.fozzy --json
fozzy ci .fozzy/traces/quic-mtls.fozzy --json
```
