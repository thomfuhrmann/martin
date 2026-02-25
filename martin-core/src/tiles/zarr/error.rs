//! Error types for `Zarr` operations

use std::path::PathBuf;

/// Errors that can occur when working with Zarr stores
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum ZarrError {
    /// IO error.
    #[error("IO error {0}: {1}")]
    IoError(#[source] std::io::Error, PathBuf),
    /// Filesystem create error
    #[error("Filesystem create error {0}: {1}")]
    FilesystemStoreCreateError(
        #[source] zarrs::filesystem::FilesystemStoreCreateError,
        PathBuf,
    ),
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
    /// Zarr WKT error
    #[error("WKT error: {0}")]
    WktError(String),
    /// Zarr attribute error
    #[error("Zarr attribute error: {0}")]
    AttributeError(String),
}
