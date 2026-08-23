//! Cache for Zarr warp-grids

use std::{borrow::Borrow, time::Duration};

use martin_tile_utils::TileCoord;
use moka::future::Cache;

use crate::tiles::zarr::utils::WarpGrid;

/// Key used for the warp cache:
///
/// The combination of tile coordinate and source id gives a unique key for the warp cache -
/// this allows fetching mutliple variables from the same Zarr store using the same warp grid
#[derive(Debug, PartialEq, Eq, Hash)]
pub(crate) struct WarpCacheKey(pub(crate) TileCoord, pub(crate) String);

impl Borrow<TileCoord> for WarpCacheKey {
    fn borrow(&self) -> &TileCoord {
        &self.0
    }
}

/// Warp cache that stores transformation from tile pixel grid to source pixel grid
#[derive(Debug, Clone)]
pub struct WarpCache {
    pub(crate) cache: Cache<WarpCacheKey, Option<WarpGrid>>,
}

impl Default for WarpCache {
    fn default() -> Self {
        Self::new(0, None, None)
    }
}

impl WarpCache {
    /// Creates a new warp cache
    #[must_use]
    pub fn new(
        max_size_bytes: u64,
        expiry: Option<Duration>,
        idle_timeout: Option<Duration>,
    ) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        let mut builder = Cache::builder()
            .name("zarr_warp_cache")
            .weigher(|_key: &WarpCacheKey, value: &Option<WarpGrid>| {
                value
                    .as_ref()
                    .map(|(grid, _)| {
                        grid.len()
                            .saturating_mul(size_of::<u32>())
                            .try_into()
                            .unwrap_or(u32::MAX)
                            + 4 * size_of::<u64>() as u32
                    })
                    .unwrap_or_default()
            })
            .max_capacity(max_size_bytes);

        if let Some(ttl) = expiry {
            builder = builder.time_to_live(ttl);
        }

        if let Some(tti) = idle_timeout {
            builder = builder.time_to_idle(tti);
        }

        Self {
            cache: builder.build(),
        }
    }
}
