use serde::{Deserialize, Serialize};
use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::api::{CodecPluginV3, PartialDecoderCapability, PartialEncoderCapability};
use zarrs::array::data_type::{Float32DataType, Float64DataType};
use zarrs::array::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, Codec, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, CodecTraitsV3, DataType, FillValue, RecommendedConcurrency,
};
use zarrs::metadata::Configuration;
use zarrs::metadata::ConfigurationSerialize;
use zarrs::metadata::v3::MetadataV3;
use zarrs_plugin::{
    PluginConfigurationInvalidError, PluginCreateError, ZarrVersion, impl_extension_aliases,
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScaleOffsetConfig {
    pub offset: f32,
    pub scale: f32,
}

impl ConfigurationSerialize for ScaleOffsetConfig {}

/// A `scale_offset` codec implementation.
#[derive(Clone, Debug)]
pub struct ScaleOffsetCodec {
    pub offset: f32,
    pub scale: f32,
}

impl ScaleOffsetCodec {
    /// Create a new `ScaleOffsetCodec` codec from configuration.
    #[must_use]
    pub fn new_with_configuration(configuration: &ScaleOffsetConfig) -> Self {
        Self {
            scale: configuration.scale,
            offset: configuration.offset,
        }
    }
}

impl_extension_aliases!(ScaleOffsetCodec, v3: "scale_offset", v2: "scale_offset");

impl CodecTraits for ScaleOffsetCodec {
    /// Return self as `Any` for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Create the codec configuration
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(
            ScaleOffsetConfig {
                scale: self.scale,
                offset: self.offset,
            }
            .into(),
        )
    }

    /// Return the partial decoder capability of the codec
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    /// Returns the partial encoder capability of the codec
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for ScaleOffsetCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn partial_decode_granularity(&self, shape: &[NonZeroU64]) -> zarrs::array::ChunkShape {
        shape.to_vec()
    }
}

impl ArrayToArrayCodecTraits for ScaleOffsetCodec {
    // Return a dynamic version of the codec
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    /// Return the encoded data type for a given decoded data type
    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(decoded_data_type.clone())
    }

    // Encode a chunk
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let scale = self.scale;
        let offset = self.offset;
        let raw_bytes = bytes.into_fixed()?;
        let mut encoded_bytes = Vec::with_capacity(raw_bytes.len());
        if data_type.is::<Float32DataType>() {
            for chunk in raw_bytes.chunks_exact(4) {
                let val = f32::from_ne_bytes(chunk.try_into().unwrap());
                let transformed = scale * (val - offset);
                encoded_bytes.extend_from_slice(&transformed.to_ne_bytes());
            }
            Ok(ArrayBytes::from(encoded_bytes))
        } else if data_type.is::<Float64DataType>() {
            for chunk in raw_bytes.chunks_exact(8) {
                let val = f64::from_ne_bytes(chunk.try_into().unwrap());
                let transformed = scale as f64 * (val - offset as f64);
                encoded_bytes.extend_from_slice(&transformed.to_ne_bytes());
            }
            Ok(ArrayBytes::from(encoded_bytes))
        } else {
            Err(CodecError::Other(format!(
                "ScaleOffsetCodec encode does not support data type: {data_type:?}"
            )))
        }
    }

    // Decode a chunk
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let scale = self.scale;
        let offset = self.offset;
        let raw_bytes = bytes.into_fixed()?;
        let mut encoded_bytes = Vec::with_capacity(raw_bytes.len());
        if data_type.is::<Float32DataType>() {
            for chunk in raw_bytes.chunks_exact(4) {
                let val = f32::from_le_bytes(chunk.try_into().unwrap());
                let transformed = val / scale + offset;
                encoded_bytes.extend_from_slice(&transformed.to_le_bytes());
            }
            Ok(ArrayBytes::from(encoded_bytes))
        } else if data_type.is::<Float64DataType>() {
            for chunk in raw_bytes.chunks_exact(8) {
                let val = f64::from_le_bytes(chunk.try_into().unwrap());
                let transformed = val / (scale as f64) + (offset as f64);
                encoded_bytes.extend_from_slice(&transformed.to_le_bytes());
            }
            Ok(ArrayBytes::from(encoded_bytes))
        } else {
            Err(CodecError::Other(format!(
                "ScaleOffsetCodec decode does not support data type: {data_type:?}"
            )))
        }
    }
}

impl CodecTraitsV3 for ScaleOffsetCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: ScaleOffsetConfig = metadata.to_configuration().map_err(|e| {
            PluginCreateError::ConfigurationInvalid(PluginConfigurationInvalidError::new(
                e.to_string(),
            ))
        })?;
        let codec = Arc::new(ScaleOffsetCodec::new_with_configuration(&configuration));
        Ok(Codec::ArrayToArray(codec))
    }
}

// Register the codec
inventory::submit! {
    CodecPluginV3::new::<ScaleOffsetCodec>()
}
