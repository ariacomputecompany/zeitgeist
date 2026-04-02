## Zeitgeist

Zeitgeist is a Rust reference runtime for an open, backend-neutral distributed inference protocol. This repository implements a strong V1 of the spec in [spec.md](/Users/deepsaint/Desktop/zeitgeist/spec.md): typed protocol objects, strict compatibility negotiation, execution planning, modular backend adapters, management APIs, and a conformance-focused test surface.

The normative implementation companion lives in [ZEITGEIST_SPEC.md](/Users/deepsaint/Desktop/zeitgeist/ZEITGEIST_SPEC.md).

## What’s implemented

- Canonical protocol objects for nodes, backends, models, kernels, tensors, cache descriptors, jobs, sessions, and events.
- A compatibility engine that classifies peers as fully compatible, conversion-compatible, API-only, solo-only, or incompatible.
- A planner that selects solo, routed, or pipeline-capable plans with explicit fallbacks.
- A Rust reference runtime with:
  - backend registry
  - model registry
  - kernel registry
  - job/session tracking
  - event stream
  - transport health reporting
  - planner decision audit history
  - automatic alternate-peer recovery and degrade-to-fallback execution
- Backend adapters for:
  - `mlx`
    - proxy-capable shape based on local `mlx_lm` execution patterns
    - synthetic mode enabled by default for deterministic certification
  - `vllm`
    - proxy-capable shape based on vLLM’s OpenAI-compatible HTTP surface
    - synthetic mode enabled by default for deterministic certification
- HTTP management and protocol endpoints:
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
  - `POST /v1/compatibility`
  - `POST /v1/plan`
  - `POST /v1/jobs`
  - `POST /v1/jobs/stream`
  - `POST /v1/jobs/{job_id}/cancel`
  - `POST /v1/tensors/roundtrip`
  - `POST /v1/cache/roundtrip`
  - `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/events`
- `GET /v1/events/stream`

Peer commands:

- `cargo run -- serve-peer --bind 127.0.0.1:9090`
- `cargo run -- serve-peer-quic --bind 127.0.0.1:9443 --cert /path/to/cert.pem --key /path/to/key.pem`
- `cargo run -- serve-peer-unix --path /tmp/zeitgeist-peer.sock`
- `cargo run -- peer-ping --addr 127.0.0.1:9090`
- `cargo run -- peer-ping-quic --addr 127.0.0.1:9443 --server-name localhost --ca-cert /path/to/cert.pem`
- `cargo run -- peer-ping-unix --path /tmp/zeitgeist-peer.sock`
- `cargo run -- peer-capabilities --addr 127.0.0.1:9090`
- `cargo run -- peer-capabilities-quic --addr 127.0.0.1:9443 --server-name localhost --ca-cert /path/to/cert.pem`
- `cargo run -- peer-execute --addr 127.0.0.1:9090 --prompt "remote execute"`
- `cargo run -- peer-execute-quic --addr 127.0.0.1:9443 --server-name localhost --ca-cert /path/to/cert.pem --prompt "remote execute"`

Peer request families now include:

- handshake
- capabilities
- compatibility
- plan
- execute job
- tensor roundtrip
- cache roundtrip

## Runtime model

The default runtime ships in deterministic synthetic mode so protocol correctness can be validated without requiring heavyweight model servers. The adapter layer is still shaped for real backends:

- `mlx` aligns to `mlx_lm.load(...)` plus `generate(...)` / `stream_generate(...)`.
- `vllm` aligns to the OpenAI-compatible server endpoints and engine/cache capability surface.

That gives us deterministic conformance locally and a clean path to live backend proxying in production deployments.

The peer transport surface now exists across:

- TCP
- QUIC with certificate-backed verification
- Unix domain sockets

## Versioning

Zeitgeist currently enforces exact protocol matching for mutating and negotiation APIs.

- Clients must send `x-zeitgeist-protocol-version`.
- The current policy is `exact_only`.
- Backwards compatibility is intentionally disabled.
- Shared-token auth can be enforced with `auth_token` in the runtime config.
- The same shared token is enforced on TCP, QUIC, and Unix peer handshakes when configured.

## Data Plane Contracts

The runtime now ships canonical binary framing implementations for:

- tensor frames
- cache blobs

Both use checksum validation and exact schema interpretation with no compatibility shims.

## Recovery

Execution is no longer single-path only.

- The runtime attempts the planned primary backend first.
- If that backend fails, alternate peers are attempted.
- If needed, the runtime degrades to fallback-compatible execution automatically.
- Successful failover is surfaced as `recovered`.

## Local setup

```bash
cargo test
uv venv .venv
uv pip install --python .venv/bin/python3 mlx mlx-lm openai
```

The repo includes a simple runtime config at [zeitgeist.toml](/Users/deepsaint/Desktop/zeitgeist/zeitgeist.toml).

## CLI

```bash
cargo run -- serve --bind 127.0.0.1:8080
cargo run -- describe --pretty
cargo run -- smoke --prompt "hello mesh"
```

`smoke` runs a deterministic in-process request so the planner and execution path can be validated without an external backend.

## Fozzy

This repo is wired to use Fozzy as the primary validation surface.

Recommended flow:

```bash
fozzy doctor --deep --scenario tests/run.pass.fozzy.json --runs 5 --seed 1337 --json
fozzy test --det --strict tests/run.pass.fozzy.json tests/memory.pass.fozzy.json tests/distributed.pass.fozzy.json --json
fozzy run tests/run.pass.fozzy.json --det --record .fozzy/traces/run.fozzy --json
fozzy trace verify .fozzy/traces/run.fozzy --strict --json
fozzy replay .fozzy/traces/run.fozzy --json
fozzy ci .fozzy/traces/run.fozzy --json
```

For live HTTP validation, run the server and point the host-backed scenario at it:

```bash
cargo run -- serve --bind 127.0.0.1:18080
fozzy run tests/host.pass.fozzy.json --det --strict --proc-backend host --fs-backend host --http-backend host --json
```

For QUIC peer regression validation:

```bash
fozzy doctor --deep --scenario tests/quic.pass.fozzy.json --runs 5 --seed 1337 --json
fozzy test --det --strict tests/quic.pass.fozzy.json --json
fozzy run tests/quic.pass.fozzy.json --det --record .fozzy/traces/quic.fozzy --json
fozzy trace verify .fozzy/traces/quic.fozzy --strict --json
fozzy replay .fozzy/traces/quic.fozzy --json
fozzy ci .fozzy/traces/quic.fozzy --json
```

## Linux Verification

For Linux-side verification, the repo can be synced into a Quilt container and exercised there.

Verified in this project:

- Quilt control plane health check succeeded.
- A Linux `prod-gui` container was created successfully with `/` as the working directory.
- The repo archive uploaded successfully into `/root/zeitgeist`.
- Linux Python 3.12 and Rust/Cargo were available.
- A modern Rust toolchain was installable with `rustup`.
- A Python venv was created successfully.
- `cargo test` passed inside the Quilt Linux container.
- Linux `vllm` wheel installation succeeded for `vllm-0.18.1-cp38-abi3-manylinux_2_31_x86_64.whl`.
- `import vllm` succeeded from the installed Linux environment.
- The `vllm.entrypoints.openai.api_server` entrypoint reached device inference and then stopped because no active GPU driver/device was available in the container.
- Real MLX inference completed through the Rust runtime on Apple Silicon.
- Real vLLM inference completed through the Rust runtime via a live Runpod RTX 4090 Linux pod and SSH tunnel.

Operational notes:

- `prod-gui` rejects custom commands at create time and starts its own GUI runtime.
- `prod-gui` also failed startup when asked to use `/workspace` as its working directory because that path did not exist in the image.
- Quilt exec on this environment expects `command` as an argv array, not a single shell string payload.
- Python package installation should use a virtualenv in the container because the base environment is PEP 668 managed.

## Notes

- The default implementation targets the spec’s narrow-V1 strategy rather than pretending every backend feature is production-complete on day one.
- `vllm` is modeled as a remote/proxy integration surface on Apple Silicon; `mlx` is locally installable and already provisioned via `uv`.
- QUIC peer transport is implemented and live-verified with TLS certificates.
