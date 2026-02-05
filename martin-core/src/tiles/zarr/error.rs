//! Error types for `Zarr` operations

use std::path::PathBuf;

use png::EncodingError;
use tiff::TiffError;

/// Errors that can occur when working with Zarr stores
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum ZarrError {
    /// IO error.
    #[error("IO error {0}: {1}")]
    IoError(#[source] std::io::Error, PathBuf),
}
