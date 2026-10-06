use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarr_cast_value::{
    FloatToIntConfig, IntToFloatConfig, MapEntry, OutOfRangeMode, RoundingMode,
    convert_slice_float_to_int, convert_slice_int_to_float,
};
use zarrs::array::codec::api::{CodecPluginV3, PartialDecoderCapability, PartialEncoderCapability};
use zarrs::array::data_type::{Float32DataType, Float64DataType, UInt16DataType};
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

/// `cast_value` codec implementation.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CastValueConfig {
    pub data_type: String,
    pub rounding: Option<RoundingMode>,
    pub out_of_range: Option<OutOfRangeMode>,
    pub scalar_map: Option<ScalarMap>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScalarMap {
    #[serde(default)]
    pub encode: Vec<(Value, Value)>,
    #[serde(default)]
    pub decode: Vec<(Value, Value)>,
}

impl ConfigurationSerialize for CastValueConfig {}

#[derive(Clone, Debug)]
pub struct CastValueCodec {
    pub data_type: DataType,
    pub rounding: Option<RoundingMode>,
    pub out_of_range: Option<OutOfRangeMode>,
    pub scalar_map: Option<ScalarMap>,
}

impl CastValueCodec {
    /// Create a new `CastValueCodec` from configuration.
    #[must_use]
    pub fn new_with_configuration(configuration: &CastValueConfig) -> Self {
        let type_str = configuration.data_type.as_str();
        let metadata = MetadataV3::new(type_str);
        let data_type = DataType::from_metadata(&metadata).expect("Should work");
        Self {
            data_type,
            rounding: configuration.rounding.clone(),
            out_of_range: configuration.out_of_range.clone(),
            scalar_map: configuration.scalar_map.clone(),
        }
    }
}

impl_extension_aliases!(CastValueCodec, v3: "cast_value", v2: "cast_value");

impl CodecTraits for CastValueCodec {
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
        if let Some(data_type) = self.data_type.name_v3().map(|val| val.to_string()) {
            Some(
                CastValueConfig {
                    data_type,
                    rounding: self.rounding.clone(),
                    out_of_range: self.out_of_range.clone(),
                    scalar_map: self.scalar_map.clone(),
                }
                .into(),
            )
        } else {
            None
        }
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

impl ArrayCodecTraits for CastValueCodec {
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

impl ArrayToArrayCodecTraits for CastValueCodec {
    // Return a dynamic version of the codec
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, _decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(self.data_type.clone())
    }

    // Encode a chunk
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType, // the decoded data type
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let raw_bytes = bytes.into_fixed()?;
        let enc_data_type = &self.data_type;
        let num_elements: usize = shape.iter().map(|s| s.get() as usize).product();
        match (data_type, enc_data_type) {
            (dec, enc)
                if dec.downcast_ref::<Float32DataType>().is_some()
                    && enc.downcast_ref::<UInt16DataType>().is_some() =>
            {
                let raw_bytes = bytemuck::cast_slice::<u8, f32>(&raw_bytes);
                let mut decoded_bytes = vec![u16::default(); num_elements];
                let map_entries: Vec<MapEntry<f32, u16>> = self
                    .scalar_map
                    .clone()
                    .map(|scalar_map| {
                        scalar_map
                            .encode
                            .into_iter()
                            .filter_map(|(k, v)| {
                                let key = value_to_f64(k)? as f32;
                                let val = serde_json::from_value(v).ok()?;
                                Some(MapEntry { src: key, tgt: val })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let config = FloatToIntConfig {
                    map_entries,
                    rounding: self.rounding.unwrap_or(RoundingMode::NearestEven),
                    out_of_range: self.out_of_range,
                };
                convert_slice_float_to_int(&raw_bytes, &mut decoded_bytes, &config)
                    .map_err(|e| CodecError::Other(e.to_string()))?;
                let byte_vec: Vec<u8> = bytemuck::pod_collect_to_vec(&decoded_bytes);
                return Ok(ArrayBytes::from(byte_vec));
            }
            (dec, enc)
                if dec.downcast_ref::<Float64DataType>().is_some()
                    && enc.downcast_ref::<UInt16DataType>().is_some() =>
            {
                let raw_bytes = bytemuck::cast_slice::<u8, f64>(&raw_bytes);
                let mut decoded_bytes = vec![u16::default(); num_elements];
                let map_entries: Vec<MapEntry<f64, u16>> = self
                    .scalar_map
                    .clone()
                    .map(|scalar_map| {
                        scalar_map
                            .encode
                            .into_iter()
                            .filter_map(|(k, v)| {
                                let key = value_to_f64(k)?;
                                let val = serde_json::from_value(v).ok()?;
                                Some(MapEntry { src: key, tgt: val })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let config = FloatToIntConfig {
                    map_entries,
                    rounding: self.rounding.unwrap_or(RoundingMode::NearestEven),
                    out_of_range: self.out_of_range,
                };
                convert_slice_float_to_int(&raw_bytes, &mut decoded_bytes, &config)
                    .map_err(|e| CodecError::Other(e.to_string()))?;
                let byte_vec: Vec<u8> = bytemuck::pod_collect_to_vec(&decoded_bytes);
                return Ok(ArrayBytes::from(byte_vec));
            }
            _ => todo!(),
        }
    }

    // Decode a chunk
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType, // the decoded data type
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let raw_bytes = bytes.into_fixed()?;
        let enc_data_type = &self.data_type;
        let num_elements: usize = shape.iter().map(|s| s.get() as usize).product();
        match (enc_data_type, data_type) {
            (enc, dec)
                if enc.downcast_ref::<UInt16DataType>().is_some()
                    && dec.downcast_ref::<Float32DataType>().is_some() =>
            {
                let raw_bytes = bytemuck::cast_slice::<u8, u16>(&raw_bytes);
                let mut decoded_bytes = vec![f32::default(); num_elements];
                let map_entries: Vec<MapEntry<u16, f32>> = self
                    .scalar_map
                    .clone()
                    .map(|scalar_map| {
                        scalar_map
                            .encode
                            .into_iter()
                            .filter_map(|(k, v)| {
                                let key = serde_json::from_value(v).ok()?;
                                let val = value_to_f64(k)? as f32;
                                Some(MapEntry { src: key, tgt: val })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let config = IntToFloatConfig {
                    map_entries,
                    rounding: self.rounding.unwrap_or(RoundingMode::NearestEven),
                };
                convert_slice_int_to_float(&raw_bytes, &mut decoded_bytes, &config)
                    .map_err(|e| CodecError::Other(e.to_string()))?;
                let byte_vec: Vec<u8> = bytemuck::pod_collect_to_vec(&decoded_bytes);
                return Ok(ArrayBytes::from(byte_vec));
            }
            (enc, dec)
                if enc.downcast_ref::<UInt16DataType>().is_some()
                    && dec.downcast_ref::<Float64DataType>().is_some() =>
            {
                let raw_bytes = bytemuck::cast_slice::<u8, u16>(&raw_bytes);
                let mut decoded_bytes = vec![f64::default(); num_elements];
                let map_entries: Vec<MapEntry<u16, f64>> = self
                    .scalar_map
                    .clone()
                    .map(|scalar_map| {
                        scalar_map
                            .encode
                            .into_iter()
                            .filter_map(|(k, v)| {
                                let key = serde_json::from_value(v).ok()?;
                                let val = value_to_f64(k)?;
                                Some(MapEntry { src: key, tgt: val })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let config = IntToFloatConfig {
                    map_entries,
                    rounding: self.rounding.unwrap_or(RoundingMode::NearestEven),
                };
                convert_slice_int_to_float(&raw_bytes, &mut decoded_bytes, &config)
                    .map_err(|e| CodecError::Other(e.to_string()))?;
                let byte_vec: Vec<u8> = bytemuck::pod_collect_to_vec(&decoded_bytes);
                return Ok(ArrayBytes::from(byte_vec));
            }
            _ => unimplemented!("Data types not supported yet"),
        }
    }
}

fn value_to_f64(val: Value) -> Option<f64> {
    match val {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.trim() {
            "NaN" => Some(f64::NAN),
            "Infinity" | "+Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            other => other.parse::<f64>().ok(),
        },
        _ => None,
    }
}

impl CodecTraitsV3 for CastValueCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: CastValueConfig = metadata.to_configuration().map_err(|e| {
            PluginCreateError::ConfigurationInvalid(PluginConfigurationInvalidError::new(
                e.to_string(),
            ))
        })?;
        let codec = Arc::new(CastValueCodec::new_with_configuration(&configuration));
        Ok(Codec::ArrayToArray(codec))
    }
}

// Register the codec
inventory::submit! {
    CodecPluginV3::new::<CastValueCodec>()
}
