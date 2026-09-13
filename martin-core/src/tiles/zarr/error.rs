//! Error types for `Zarr` operations

use std::path::PathBuf;

use image::ImageError;

/// Errors that can occur when working with Zarr stores
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum ZarrError {
    /// IO error.
    #[error("IO error {0}: {1}")]
    IoError(#[source] std::io::Error, PathBuf),
    /// Filesystem create error
    #[error("Object store create error {0}: {1}")]
    OjbectStoreError(#[source] object_store::Error, PathBuf),
    /// Zarr array error
    #[error("Zarr array error: {0}")]
    ArrayError(#[source] zarrs::array::ArrayError),
    /// Zarr array error
    #[error("Zarr array create error: {0}")]
    ArrayCreateError(#[source] zarrs::array::ArrayCreateError),
    /// Zarr group create error
    #[error("Zarr group create error: {0}")]
    GroupCreateError(#[source] zarrs::group::GroupCreateError),
    /// Zarr node path error
    #[error("Zarr node path error: {0}")]
    NodePathError(#[source] zarrs::node::NodePathError),
    /// Zarr node create error
    #[error("Zarr node create error: {0}")]
    NodeCreateError(#[source] zarrs::node::NodeCreateError),
    /// Zarr attribute error
    #[error("Zarr attribute error: {0}")]
    AttributeError(String),
    /// Chrono parse error
    #[error("Chrono parsing error: {0}")]
    ParseError(chrono::ParseError),
    /// Projection creation error
    #[error("Projection create error: {0}")]
    ProjCreateError(proj::ProjCreateError),
    /// Projection error
    #[error("Conversion error: {0}")]
    ProjError(proj::ProjError),
    /// Time error
    #[error("Time attribute error: {0}")]
    TimeError(String),
    /// Dimension error
    #[error("Dimension error: {0}")]
    DimensionError(String),
    /// Encode error
    #[error("Encode error: {0}")]
    EncodeError(std::io::Error),
    /// Decode error
    #[error("Decode error: {0}")]
    DecodeError(String),
    /// Cast error
    #[error("Cast error: {0}")]
    CastError(String),
    /// Warp error
    #[error("Warp error: {0}")]
    WarpError(String),
    /// Image error
    #[error("Image error: {0}")]
    ImageError(ImageError),
}
