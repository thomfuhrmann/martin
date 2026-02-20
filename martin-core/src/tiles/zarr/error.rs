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
}
