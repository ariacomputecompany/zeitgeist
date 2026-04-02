use crate::types::{CacheBlob, TensorFrame};
use anyhow::{Result, anyhow};

const TENSOR_MAGIC: &[u8; 5] = b"ZGTN1";
const CACHE_MAGIC: &[u8; 5] = b"ZGKC1";

pub fn encode_tensor_frame(frame: &TensorFrame) -> Result<Vec<u8>> {
    frame.validate()?;
    encode_with_magic(TENSOR_MAGIC, frame)
}

pub fn decode_tensor_frame(bytes: &[u8]) -> Result<TensorFrame> {
    let frame: TensorFrame = decode_with_magic(TENSOR_MAGIC, bytes)?;
    frame.validate()?;
    Ok(frame)
}

pub fn encode_cache_blob(blob: &CacheBlob) -> Result<Vec<u8>> {
    blob.validate()?;
    encode_with_magic(CACHE_MAGIC, blob)
}

pub fn decode_cache_blob(bytes: &[u8]) -> Result<CacheBlob> {
    let blob: CacheBlob = decode_with_magic(CACHE_MAGIC, bytes)?;
    blob.validate()?;
    Ok(blob)
}

fn encode_with_magic<T: serde::Serialize>(magic: &[u8; 5], payload: &T) -> Result<Vec<u8>> {
    let body = serde_json::to_vec(payload)?;
    let mut bytes = Vec::with_capacity(magic.len() + 4 + body.len());
    bytes.extend_from_slice(magic);
    bytes.extend_from_slice(&(body.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

fn decode_with_magic<T: serde::de::DeserializeOwned>(magic: &[u8; 5], bytes: &[u8]) -> Result<T> {
    if bytes.len() < 9 {
        return Err(anyhow!("frame too short"));
    }
    if &bytes[..5] != magic {
        return Err(anyhow!("frame magic mismatch"));
    }
    let len = u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
    if bytes.len() != 9 + len {
        return Err(anyhow!("frame length mismatch"));
    }
    Ok(serde_json::from_slice(&bytes[9..])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use uuid::Uuid;

    fn tensor_frame() -> TensorFrame {
        TensorFrame {
            envelope: TensorEnvelope {
                tensor_id: "t1".into(),
                op_context_id: "ctx1".into(),
                session_id: Uuid::nil(),
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
                checksum: TensorFrame::checksum_hex(&[1, 2, 3, 4]),
                sequence_number: 7,
            },
            payload: vec![1, 2, 3, 4],
        }
    }

    #[test]
    fn tensor_frame_roundtrips() {
        let encoded = encode_tensor_frame(&tensor_frame()).unwrap();
        let decoded = decode_tensor_frame(&encoded).unwrap();
        assert_eq!(decoded.payload, vec![1, 2, 3, 4]);
        assert_eq!(decoded.envelope.sequence_number, 7);
    }

    #[test]
    fn cache_blob_roundtrips() {
        let blob = CacheBlob {
            cache_id: "cache1".into(),
            session_id: Uuid::nil(),
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
            token_count: 16,
            checksum: CacheBlob::checksum_hex(&[8, 6, 7, 5, 3, 0, 9]),
            payload: vec![8, 6, 7, 5, 3, 0, 9],
        };
        let encoded = encode_cache_blob(&blob).unwrap();
        let decoded = decode_cache_blob(&encoded).unwrap();
        assert_eq!(decoded.token_count, 16);
        assert_eq!(decoded.payload.len(), 7);
    }
}
