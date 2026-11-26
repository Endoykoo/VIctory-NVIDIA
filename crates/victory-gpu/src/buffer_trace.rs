//! Optional address-only trace of buffer readbacks and visible consumers.
//! `VICTORY_TRACE_BUFFER_READS=probe` traces only the 90 MiB allocation class
//! so the external watchpoint sampler can find it with less logging.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

static FIRST: AtomicU64 = AtomicU64::new(u64::MAX);
static PREVIOUS_DRAWS: AtomicU64 = AtomicU64::new(0);

fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_TRACE_BUFFER_READS").as_deref() {
        Ok("1") => 1,
        Ok("probe") => 2,
        _ => 0,
    })
}

pub(crate) fn presented(frame: u64, draws: u64) {
    if mode() == 0 {
        return;
    }
    let first = FIRST.load(Relaxed);
    if first == u64::MAX && draws.saturating_sub(PREVIOUS_DRAWS.swap(draws, Relaxed)) >= 500 {
        FIRST.store(frame, Relaxed);
        tracing::debug!(frame, "buffer trace started");
    } else if first != u64::MAX && frame.saturating_sub(first) == 100 {
        tracing::debug!(frame, "buffer trace complete");
    }
}

pub(crate) fn range(kind: &str, address: usize, bytes: usize, base: u64, capacity: u64) {
    let first = FIRST.load(Relaxed);
    if first == u64::MAX || bytes == 0 {
        return;
    }
    let frame = crate::queue::presented();
    if frame.wrapping_sub(first) < 100 && (mode() == 1 || kind == "readback" && capacity == 90 << 20)
    {
        tracing::debug!(frame, kind, address, bytes, base, capacity, "buffer access");
    }
}
