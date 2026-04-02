# Zeitgeist Reference Runtime

This document is the normative implementation companion to [spec.md](/Users/deepsaint/Desktop/zeitgeist/spec.md). It describes the exact behavior implemented in this repository today. There are no backwards-compatibility guarantees.

## Versioning

- Protocol version: `0.1.0`
- Compatibility mode: `exact_only`
- Backwards compatibility: disabled
- Mutating and negotiation endpoints require `x-zeitgeist-protocol-version: 0.1.0`

## Authentication

- Auth modes modeled: `none`, `shared_token`, `mutual_tls`, `signed_node_identity`, `backend_signed_attestation`, `extension`
- Enforced auth mode in this repo: `shared_token` when `auth_token` is configured
- Shared token header:
  - `Authorization: Bearer <token>`
- Peer transport auth field:
  - raw `auth_token` inside the exact-version request envelope

## Management and Protocol HTTP Surface

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

## Peer Surface

- `serve-peer` starts the node-to-node TCP listener
- `serve-peer-quic` starts the node-to-node QUIC listener and requires a certificate/key pair
- `serve-peer-unix` starts the node-to-node Unix domain socket listener
- `peer-ping` performs an exact-version authenticated handshake
- `peer-ping-quic` performs an exact-version authenticated QUIC handshake with certificate validation
- `peer-ping-unix` performs an exact-version authenticated Unix-socket handshake
- `peer-capabilities-quic` queries capabilities over QUIC
- `peer-execute-quic` executes a remote job over QUIC
- implemented peer requests:
  - `handshake`
  - `capabilities`
  - `compatibility`
  - `plan`
  - `execute_job`
  - `tensor_roundtrip`
  - `cache_roundtrip`

Framing:

- magic: `ZGP1`
- exact-length framed JSON
- exact protocol version enforcement
- shared-token auth enforcement when configured
- transport variants:
  - TCP
  - QUIC with certificate-backed verification
  - Unix domain sockets

## Execution Semantics

- Job submission plans execution against registered backends.
- The primary backend is attempted first.
- If the primary backend fails, alternate peers are attempted.
- If no alternate peer was explicitly requested but fallback-compatible backends exist, the runtime degrades to fallback execution automatically.
- Successful failover is surfaced as `recovered`.
- Streaming execution emits SSE chunks and can be cancelled through the cancellation endpoint.
- Peer transport can execute a remote job request and return a full execution record.

## Implemented Data Plane Contracts

### Tensor Frames

- Canonical JSON schema plus canonical binary frame envelope
- Magic prefix: `ZGTN1`
- Length-prefixed payload
- SHA-256 checksum validation
- No compatibility shims

### Cache Blobs

- Canonical JSON schema plus canonical binary frame envelope
- Magic prefix: `ZGKC1`
- Length-prefixed payload
- SHA-256 checksum validation
- No compatibility shims

## Backend Model

- Synthetic backends for deterministic certification
- MLX-shaped backend interface
- vLLM OpenAI-compatible proxy-shaped backend interface
- Backend failover and degrade-to-fallback implemented in runtime orchestration

## Verification Surfaces

- Rust unit and integration tests
- Fozzy deterministic doctor/test/run/replay/ci
- Fozzy deterministic QUIC doctor/test/run/replay/ci
- Fozzy explore/fuzz/profile/memory/corpus flows
- Live host-backed HTTP verification
- Live QUIC peer handshake/capabilities/remote execution verification
- Live tensor/cache roundtrip verification
- Live auth enforcement verification
- Live SSE streaming verification
- Live MLX adapter inference verification on Apple Silicon
- Live remote vLLM inference verification through a Linux GPU pod and SSH tunnel
- Linux-side Quilt container verification path
- Verified Linux `cargo test` pass in Quilt
- Verified Linux `vllm` install/import for x86_64 manylinux
- Verified Linux `vllm` OpenAI entrypoint reaches device inference and currently stops without an active GPU/device
