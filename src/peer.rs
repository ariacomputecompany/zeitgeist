use crate::{codec, runtime::Runtime, types::*};
use anyhow::{anyhow, Context, Result};
use quinn::{
    crypto::rustls::{QuicClientConfig, QuicServerConfig},
    ClientConfig, Endpoint, ServerConfig,
};
use serde::{Deserialize, Serialize};
use std::{net::SocketAddr, path::Path, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream, UnixListener, UnixStream},
};

const PEER_MAGIC: &[u8; 4] = b"ZGP1";
const PEER_ALPN: &[u8] = b"zeitgeist-peer/0.1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PeerRequest {
    Handshake { protocol_version: String, auth_token: Option<String>, node_id: String },
    Capabilities { protocol_version: String, auth_token: Option<String> },
    Compatibility { protocol_version: String, auth_token: Option<String>, request: CompatibilityRequest },
    Plan { protocol_version: String, auth_token: Option<String>, request: JobRequest },
    ExecuteJob { protocol_version: String, auth_token: Option<String>, request: JobRequest },
    TensorRoundTrip { protocol_version: String, auth_token: Option<String>, frame: TensorFrame },
    CacheRoundTrip { protocol_version: String, auth_token: Option<String>, blob: CacheBlob },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PeerResponse {
    HandshakeAccepted { protocol_version: String, node: NodeIdentity, compatibility_mode: String },
    Capabilities { snapshot: CapabilitySnapshot },
    Compatibility { report: CompatibilityReport },
    Plan { plan: JobPlan },
    ExecuteJob { record: JobRecord },
    TensorRoundTrip { byte_len: usize, checksum: String, sequence_number: u64 },
    CacheRoundTrip { byte_len: usize, checksum: String, token_count: u32 },
    Error { code: String, message: String },
}

pub async fn serve(runtime: Runtime, bind: &str) -> Result<()> {
    let listener = TcpListener::bind(bind).await?;
    loop {
        let (stream, _) = listener.accept().await?;
        let runtime = runtime.clone();
        tokio::spawn(async move {
            let _ = handle_connection(runtime, stream).await;
        });
    }
}

pub async fn serve_unix(runtime: Runtime, path: &str) -> Result<()> {
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path)?;
    loop {
        let (stream, _) = listener.accept().await?;
        let runtime = runtime.clone();
        tokio::spawn(async move {
            let _ = handle_unix_connection(runtime, stream).await;
        });
    }
}

pub async fn serve_quic(runtime: Runtime, bind: &str, cert_path: &Path, key_path: &Path) -> Result<()> {
    let cert_pem = std::fs::read(cert_path)
        .with_context(|| format!("failed to read QUIC certificate {}", cert_path.display()))?;
    let key_pem =
        std::fs::read(key_path).with_context(|| format!("failed to read QUIC key {}", key_path.display()))?;
    let bind_addr: SocketAddr = bind.parse().with_context(|| format!("invalid QUIC bind address {bind}"))?;
    serve_quic_with_material(runtime, bind_addr, &cert_pem, &key_pem).await
}

pub async fn send(addr: &str, request: &PeerRequest) -> Result<PeerResponse> {
    let mut stream = TcpStream::connect(addr)
        .await
        .with_context(|| format!("failed to connect to peer {addr}"))?;
    write_frame(&mut stream, request).await?;
    read_frame(&mut stream).await
}

pub async fn send_unix(path: &str, request: &PeerRequest) -> Result<PeerResponse> {
    let mut stream = UnixStream::connect(path)
        .await
        .with_context(|| format!("failed to connect to unix peer {path}"))?;
    write_frame(&mut stream, request).await?;
    read_frame(&mut stream).await
}

pub async fn send_quic(addr: &str, server_name: &str, ca_cert_path: &Path, request: &PeerRequest) -> Result<PeerResponse> {
    let ca_cert_pem = std::fs::read(ca_cert_path)
        .with_context(|| format!("failed to read QUIC CA certificate {}", ca_cert_path.display()))?;
    let addr: SocketAddr = addr.parse().with_context(|| format!("invalid QUIC peer address {addr}"))?;
    send_quic_with_ca(addr, server_name, &ca_cert_pem, request).await
}

async fn handle_connection(runtime: Runtime, mut stream: TcpStream) -> Result<()> {
    handle_peer_request(runtime, &mut stream).await
}

async fn handle_unix_connection(runtime: Runtime, mut stream: UnixStream) -> Result<()> {
    handle_peer_request(runtime, &mut stream).await
}

async fn serve_quic_with_material(runtime: Runtime, bind: SocketAddr, cert_pem: &[u8], key_pem: &[u8]) -> Result<()> {
    let server_config = make_quic_server_config(cert_pem, key_pem)?;
    let endpoint = Endpoint::server(server_config, bind)?;
    while let Some(incoming) = endpoint.accept().await {
        let runtime = runtime.clone();
        tokio::spawn(async move {
            let Ok(connection) = incoming.await else {
                return;
            };
            let Ok((mut send, mut recv)) = connection.accept_bi().await else {
                return;
            };
            let _ = handle_peer_request_split(runtime, &mut recv, &mut send).await;
            let _ = send.flush().await;
            let _ = send.finish();
            let _ = connection.closed().await;
        });
    }
    Ok(())
}

async fn send_quic_with_ca(
    addr: SocketAddr,
    server_name: &str,
    ca_cert_pem: &[u8],
    request: &PeerRequest,
) -> Result<PeerResponse> {
    let client_addr: SocketAddr = "0.0.0.0:0".parse().unwrap();
    let mut endpoint = Endpoint::client(client_addr)?;
    endpoint.set_default_client_config(make_quic_client_config(ca_cert_pem)?);
    let connection = endpoint
        .connect(addr, server_name)
        .with_context(|| format!("failed to start QUIC connection to {addr}"))?
        .await
        .with_context(|| format!("failed to establish QUIC connection to {addr}"))?;
    let (mut send, mut recv) = connection.open_bi().await.context("failed to open QUIC stream")?;
    write_frame_to(&mut send, request).await?;
    send.finish().context("failed to finish QUIC request stream")?;
    let response = read_frame_from(&mut recv).await;
    connection.close(0u32.into(), b"done");
    endpoint.wait_idle().await;
    response
}

async fn handle_peer_request<S>(runtime: Runtime, stream: &mut S) -> Result<()>
where
    S: AsyncReadExt + AsyncWriteExt + Unpin,
{
    let request: PeerRequest = read_frame(stream).await?;
    let response = route_peer_request(&runtime, request).await;
    write_frame(stream, &response).await?;
    Ok(())
}

async fn handle_peer_request_split<R, W>(runtime: Runtime, reader: &mut R, writer: &mut W) -> Result<()>
where
    R: AsyncReadExt + Unpin,
    W: AsyncWriteExt + Unpin,
{
    let request: PeerRequest = read_frame_from(reader).await?;
    let response = route_peer_request(&runtime, request).await;
    write_frame_to(writer, &response).await?;
    Ok(())
}

async fn route_peer_request(runtime: &Runtime, request: PeerRequest) -> PeerResponse {
    match request {
        PeerRequest::Handshake { protocol_version, auth_token, node_id: _ } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => PeerResponse::HandshakeAccepted {
                    protocol_version: runtime.protocol_version().to_string(),
                    node: runtime.capabilities().node,
                    compatibility_mode: crate::runtime::VERSION_POLICY.into(),
                },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::Capabilities { protocol_version, auth_token } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => PeerResponse::Capabilities { snapshot: runtime.capabilities() },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::Compatibility { protocol_version, auth_token, request } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => PeerResponse::Compatibility { report: runtime.compatibility(&request) },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::Plan { protocol_version, auth_token, request } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => match runtime.plan(&request) {
                    Ok(plan) => PeerResponse::Plan { plan },
                    Err(error) => PeerResponse::Error { code: "plan_error".into(), message: error.to_string() },
                },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::ExecuteJob { protocol_version, auth_token, request } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => match runtime.submit_job(request).await {
                    Ok(record) => PeerResponse::ExecuteJob { record },
                    Err(error) => PeerResponse::Error { code: "execute_error".into(), message: error.to_string() },
                },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::TensorRoundTrip { protocol_version, auth_token, frame } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => match codec::encode_tensor_frame(&frame)
                    .and_then(|bytes| codec::decode_tensor_frame(&bytes).map(|decoded| (bytes, decoded)))
                {
                    Ok((bytes, decoded)) => PeerResponse::TensorRoundTrip {
                        byte_len: bytes.len(),
                        checksum: decoded.envelope.checksum,
                        sequence_number: decoded.envelope.sequence_number,
                    },
                    Err(error) => PeerResponse::Error { code: "tensor_roundtrip_error".into(), message: error.to_string() },
                },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
        PeerRequest::CacheRoundTrip { protocol_version, auth_token, blob } => {
            match validate_peer_request(runtime, &protocol_version, auth_token.as_deref()) {
                Ok(()) => match codec::encode_cache_blob(&blob)
                    .and_then(|bytes| codec::decode_cache_blob(&bytes).map(|decoded| (bytes, decoded)))
                {
                    Ok((bytes, decoded)) => PeerResponse::CacheRoundTrip {
                        byte_len: bytes.len(),
                        checksum: decoded.checksum,
                        token_count: decoded.token_count,
                    },
                    Err(error) => PeerResponse::Error { code: "cache_roundtrip_error".into(), message: error.to_string() },
                },
                Err((code, message)) => PeerResponse::Error { code, message },
            }
        }
    }
}

fn make_quic_server_config(cert_pem: &[u8], key_pem: &[u8]) -> Result<ServerConfig> {
    let cert_chain = load_certs(cert_pem)?;
    let key = load_private_key(key_pem)?;
    let mut crypto = quinn::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, key)
        .context("failed to build QUIC TLS server config")?;
    crypto.alpn_protocols = vec![PEER_ALPN.to_vec()];
    let mut server_config = ServerConfig::with_crypto(Arc::new(
        QuicServerConfig::try_from(crypto).context("failed to adapt rustls QUIC server config")?,
    ));
    server_config.transport = Arc::new(quinn::TransportConfig::default());
    Ok(server_config)
}

fn make_quic_client_config(ca_cert_pem: &[u8]) -> Result<ClientConfig> {
    let mut roots = quinn::rustls::RootCertStore::empty();
    for cert in load_certs(ca_cert_pem)? {
        roots
            .add(cert)
            .context("failed to add QUIC CA certificate to trust store")?;
    }
    let mut crypto = quinn::rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    crypto.alpn_protocols = vec![PEER_ALPN.to_vec()];
    Ok(ClientConfig::new(Arc::new(
        QuicClientConfig::try_from(crypto).context("failed to adapt rustls QUIC client config")?,
    )))
}

fn load_certs(mut cert_pem: &[u8]) -> Result<Vec<quinn::rustls::pki_types::CertificateDer<'static>>> {
    rustls_pemfile::certs(&mut cert_pem)
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse PEM certificate chain")
}

fn load_private_key(mut key_pem: &[u8]) -> Result<quinn::rustls::pki_types::PrivateKeyDer<'static>> {
    rustls_pemfile::private_key(&mut key_pem)
        .context("failed to parse PEM private key")?
        .ok_or_else(|| anyhow!("no private key found in PEM payload"))
}

fn validate_peer_request(
    runtime: &Runtime,
    protocol_version: &str,
    auth_token: Option<&str>,
) -> std::result::Result<(), (String, String)> {
    if protocol_version != runtime.protocol_version() {
        return Err((
            "protocol_version_mismatch".into(),
            format!("received {}, required {}", protocol_version, runtime.protocol_version()),
        ));
    }
    if let Some(expected) = runtime.auth_token() {
        if auth_token != Some(expected) {
            return Err(("auth_failed".into(), "shared token mismatch".into()));
        }
    }
    Ok(())
}

async fn write_frame<S, T>(stream: &mut S, value: &T) -> Result<()>
where
    S: AsyncWriteExt + Unpin,
    T: Serialize,
{
    write_frame_to(stream, value).await
}

async fn write_frame_to<W, T>(writer: &mut W, value: &T) -> Result<()>
where
    W: AsyncWriteExt + Unpin,
    T: Serialize,
{
    let payload = serde_json::to_vec(value)?;
    writer.write_all(PEER_MAGIC).await?;
    writer.write_all(&(payload.len() as u32).to_le_bytes()).await?;
    writer.write_all(&payload).await?;
    Ok(())
}

async fn read_frame<S, T>(stream: &mut S) -> Result<T>
where
    S: AsyncReadExt + Unpin,
    T: for<'de> Deserialize<'de>,
{
    read_frame_from(stream).await
}

async fn read_frame_from<R, T>(reader: &mut R) -> Result<T>
where
    R: AsyncReadExt + Unpin,
    T: for<'de> Deserialize<'de>,
{
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).await?;
    if &magic != PEER_MAGIC {
        return Err(anyhow!("peer frame magic mismatch"));
    }
    let mut len = [0u8; 4];
    reader.read_exact(&mut len).await?;
    let len = u32::from_le_bytes(len) as usize;
    let mut payload = vec![0u8; len];
    reader.read_exact(&mut payload).await?;
    Ok(serde_json::from_slice(&payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{backend::default_backends, config};
    use rcgen::generate_simple_self_signed;

    fn test_runtime() -> Runtime {
        Runtime::new(
            config::node_identity(&Default::default()),
            default_backends(),
            config::models(),
            config::kernels(),
            Some("peer-secret".into()),
        )
    }

    async fn spawn_peer_server(runtime: Runtime) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let runtime = runtime.clone();
                tokio::spawn(async move {
                    let _ = super::handle_connection(runtime, stream).await;
                });
            }
        });
        addr.to_string()
    }

    async fn spawn_unix_peer_server(runtime: Runtime, path: &str) {
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let runtime = runtime.clone();
                tokio::spawn(async move {
                    let _ = super::handle_unix_connection(runtime, stream).await;
                });
            }
        });
    }

    fn quic_material() -> (Vec<u8>, Vec<u8>) {
        let cert = generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let cert_pem = cert.cert.pem();
        let key_pem = cert.key_pair.serialize_pem();
        (cert_pem.into_bytes(), key_pem.into_bytes())
    }

    async fn spawn_quic_peer_server(runtime: Runtime) -> (String, Vec<u8>) {
        let (cert_pem, key_pem) = quic_material();
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = socket.local_addr().unwrap();
        drop(socket);
        tokio::spawn({
            let cert_pem = cert_pem.clone();
            let key_pem = key_pem.clone();
            async move {
                let _ = serve_quic_with_material(runtime, addr, &cert_pem, &key_pem).await;
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        (addr.to_string(), cert_pem)
    }

    #[tokio::test]
    async fn peer_handshake_accepts_exact_version_and_auth() {
        let addr = spawn_peer_server(test_runtime()).await;
        let response = send(
            &addr,
            &PeerRequest::Handshake {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                node_id: "client-node".into(),
            },
        )
        .await
        .unwrap();

        match response {
            PeerResponse::HandshakeAccepted { protocol_version, .. } => assert_eq!(protocol_version, "0.1.0"),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn peer_handshake_rejects_bad_auth() {
        let addr = spawn_peer_server(test_runtime()).await;
        let response = send(
            &addr,
            &PeerRequest::Handshake {
                protocol_version: "0.1.0".into(),
                auth_token: Some("wrong".into()),
                node_id: "client-node".into(),
            },
        )
        .await
        .unwrap();

        match response {
            PeerResponse::Error { code, .. } => assert_eq!(code, "auth_failed"),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn peer_tensor_roundtrip_works() {
        let addr = spawn_peer_server(test_runtime()).await;
        let response = send(
            &addr,
            &PeerRequest::TensorRoundTrip {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                frame: TensorFrame {
                    envelope: TensorEnvelope {
                        tensor_id: "t1".into(),
                        op_context_id: "ctx1".into(),
                        session_id: uuid::Uuid::nil(),
                        role: "activation".into(),
                        shape: vec![1, 4],
                        dtype: DType::F16,
                        layout: TensorLayout::RowMajorContiguous,
                        quantization: QuantizationDescriptor {
                            format: QuantFormat::None,
                            group_size: None,
                            scale_dtype: None,
                            zero_point_dtype: None,
                            packing_layout: None,
                            calibration: None,
                        },
                        compression: false,
                        checksum: TensorFrame::checksum_hex(b"peer"),
                        sequence_number: 11,
                    },
                    payload: b"peer".to_vec(),
                },
            },
        )
        .await
        .unwrap();

        match response {
            PeerResponse::TensorRoundTrip { sequence_number, .. } => assert_eq!(sequence_number, 11),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn peer_cache_roundtrip_works() {
        let addr = spawn_peer_server(test_runtime()).await;
        let response = send(
            &addr,
            &PeerRequest::CacheRoundTrip {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                blob: CacheBlob {
                    cache_id: "cache1".into(),
                    session_id: uuid::Uuid::nil(),
                    model_id: "llama-3.2-3b-instruct".into(),
                    descriptor: CacheDescriptor {
                        version: "zgc-1".into(),
                        dtype: DType::F16,
                        layout: TensorLayout::RowMajorContiguous,
                        head_grouping: "grouped-query".into(),
                        rope_state: PositionEncoding::Rope,
                        sequence_indexing: "absolute".into(),
                        eviction: "lru".into(),
                        compression: None,
                        transferable: true,
                    },
                    token_count: 8,
                    checksum: CacheBlob::checksum_hex(&[1, 2, 3, 4]),
                    payload: vec![1, 2, 3, 4],
                },
            },
        )
        .await
        .unwrap();

        match response {
            PeerResponse::CacheRoundTrip { token_count, .. } => assert_eq!(token_count, 8),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn peer_execute_job_works() {
        let addr = spawn_peer_server(test_runtime()).await;
        let response = send(
            &addr,
            &PeerRequest::ExecuteJob {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                request: JobRequest {
                    model_id: "llama-3.2-3b-instruct".into(),
                    job_type: JobType::DistributedShardExecution,
                    prompt: "remote execute".into(),
                    session_id: None,
                    preferred_backends: vec!["mlx".into()],
                    max_tokens: 8,
                    temperature: 0.0,
                    determinism: DeterminismPolicy {
                        strict_correctness: true,
                        deterministic: true,
                        low_latency: true,
                        high_availability: true,
                    },
                },
            },
        )
        .await
        .unwrap();

        match response {
            PeerResponse::ExecuteJob { record } => {
                assert!(matches!(record.status, JobStatus::Completed | JobStatus::Recovered));
                assert!(record.result.is_some());
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn unix_peer_handshake_works() {
        let path = format!("/tmp/zeitgeist-peer-{}.sock", uuid::Uuid::new_v4());
        spawn_unix_peer_server(test_runtime(), &path).await;
        let response = send_unix(
            &path,
            &PeerRequest::Handshake {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                node_id: "unix-client".into(),
            },
        )
        .await
        .unwrap();
        match response {
            PeerResponse::HandshakeAccepted { protocol_version, .. } => assert_eq!(protocol_version, "0.1.0"),
            other => panic!("unexpected response: {other:?}"),
        }
        let _ = std::fs::remove_file(path);
    }

    #[tokio::test]
    async fn quic_peer_handshake_works() {
        let (addr, cert_pem) = spawn_quic_peer_server(test_runtime()).await;
        let response = send_quic_with_ca(
            addr.parse().unwrap(),
            "localhost",
            &cert_pem,
            &PeerRequest::Handshake {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                node_id: "quic-client".into(),
            },
        )
        .await
        .unwrap();
        match response {
            PeerResponse::HandshakeAccepted { protocol_version, .. } => assert_eq!(protocol_version, "0.1.0"),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[tokio::test]
    async fn quic_peer_execute_job_works() {
        let (addr, cert_pem) = spawn_quic_peer_server(test_runtime()).await;
        let response = send_quic_with_ca(
            addr.parse().unwrap(),
            "localhost",
            &cert_pem,
            &PeerRequest::ExecuteJob {
                protocol_version: "0.1.0".into(),
                auth_token: Some("peer-secret".into()),
                request: JobRequest {
                    model_id: "llama-3.2-3b-instruct".into(),
                    job_type: JobType::DistributedShardExecution,
                    prompt: "quic execute".into(),
                    session_id: None,
                    preferred_backends: vec!["mlx".into()],
                    max_tokens: 8,
                    temperature: 0.0,
                    determinism: DeterminismPolicy {
                        strict_correctness: true,
                        deterministic: true,
                        low_latency: true,
                        high_availability: true,
                    },
                },
            },
        )
        .await
        .unwrap();
        match response {
            PeerResponse::ExecuteJob { record } => assert!(record.result.is_some()),
            other => panic!("unexpected response: {other:?}"),
        }
    }
}
