//! Off by default (`VICTORY_MERGE_BATCH`): the title's changes under flights' shader writes to ONE copy in
//! the card's memory, applied in one compute dispatch instead of one copy
//! command per changed run (`VICTORY_MERGE_BATCH=<hex title address>:<mode>`;
//! off unless set).
//!
//! `Shadow::merge_under_writes` finds, page by page, the runs of bytes the
//! title changed in a range a flight's shaders may be writing, and puts each
//! run into the copy's CPU side; each run then goes to the card side as a
//! copy command of its own (`Host::card_before`), and past `VICTORY_MERGE_RUNS`
//! runs a submission the flights are settled instead. Here the same runs
//! become dword records (dword, value, byte mask) that one dispatch applies:
//! `dst = (dst & !mask) | (value & mask)`, so every byte the title did not
//! change keeps what the card side holds, shader writes included. Each dword
//! has one record, so no two threads write the same dword.
//!
//! Modes:
//! * `count`: nothing changes; the runs of that copy are counted per
//!   submission, those past the cap too (the copy commands it takes).
//! * `check`: nothing changes for the game (the copy commands run as before);
//!   the dispatch is applied to a twin of the card side taken right before,
//!   and both the card side before and the twin after are read back. The CPU
//!   rebuilds what the copy commands put there from the recorded runs (each
//!   run's bytes over the bytes before) and compares it with the twin byte for
//!   byte; it also counts twin bytes that differ from before outside the runs
//!   (must be none: those are bytes the title did not change).
//! * `on`: the dispatch replaces the copy's per-run copy commands, and the
//!   copy's merge has no cap.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Count,
    Check,
    On,
}

/// The copy: at a title address, or (`len:<hex bytes>`) the first copy of
/// that length whose merge overruns the cap `LOCK_AFTER` times (title buffers
/// do not keep their address from run to run).
#[derive(Clone, Copy)]
enum Target {
    At(u64),
    Len(u64),
    /// Every copy in the card's memory (`all:on`).
    All,
}

const LOCK_AFTER: u32 = 3;
static LOCKED: AtomicU64 = AtomicU64::new(0);
static OVERRUNS: Mutex<Vec<(u64, u32)>> = Mutex::new(Vec::new());

fn flag() -> Option<(Target, Mode)> {
    static FLAG: OnceLock<Option<(Target, Mode)>> = OnceLock::new();
    *FLAG.get_or_init(|| {
        let v = std::env::var("VICTORY_MERGE_BATCH").ok()?;
        let v = v.trim();
        let (target, rest) = if let Some(mode) = v.strip_prefix("all:") {
            (Target::All, mode)
        } else { match v.strip_prefix("len:") {
            Some(r) => {
                let (len, mode) = r.split_once(':').unwrap_or((r, "count"));
                (Target::Len(u64::from_str_radix(len.trim_start_matches("0x"), 16).ok()?), mode)
            }
            None => {
                let (at, mode) = v.split_once(':').unwrap_or((v, "count"));
                (Target::At(u64::from_str_radix(at.trim_start_matches("0x"), 16).ok()?), mode)
            }
        } };
        let mode = match rest {
            "check" => Mode::Check,
            "on" => Mode::On,
            _ => Mode::Count,
        };
        tracing::info!(
            target = match target { Target::At(a) => format!("copy at {a:#x}"), Target::Len(l) => format!("first copy of {l:#x} bytes to overrun the cap {LOCK_AFTER} times"), Target::All => "every copy in the card's memory".to_string() },
            mode = match mode { Mode::Count => "count", Mode::Check => "check", Mode::On => "on" },
            "VICTORY_MERGE_BATCH"
        );
        Some((target, mode))
    })
}

/// The mode for the copy at `start` of `len` bytes whose merge has `runs`
/// changed runs with `left` of the cap left, if it is the one named.
pub(crate) fn mode_for(start: u64, len: u64, runs: usize, left: usize) -> Option<Mode> {
    let (target, mode) = flag()?;
    // Nothing is batched or checked unless the pass can run: the copy
    // commands stay (`ready`).
    if mode != Mode::Count && !ready() {
        return None;
    }
    match target {
        Target::At(a) => (a == start).then_some(mode),
        // (Every copy: only the batch itself; the check is for one copy.)
        Target::All => (mode == Mode::On).then_some(mode),
        Target::Len(l) => {
            let locked = LOCKED.load(Relaxed);
            if locked != 0 {
                return (locked == start).then_some(mode);
            }
            if len == l && runs > left {
                let mut o = OVERRUNS.lock().unwrap_or_else(|e| e.into_inner());
                let n = match o.iter_mut().find(|e| e.0 == start) {
                    Some(e) => {
                        e.1 += 1;
                        e.1
                    }
                    None => {
                        o.push((start, 1));
                        1
                    }
                };
                if n >= LOCK_AFTER && LOCKED.compare_exchange(0, start, Relaxed, Relaxed).is_ok() {
                    tracing::info!(copy = format_args!("{start:#x}"), "VICTORY_MERGE_BATCH: the copy is chosen");
                }
            }
            None
        }
    }
}

/// The copy named (once chosen), if any.
pub(crate) fn named() -> Option<(u64, Mode)> {
    let (target, mode) = flag()?;
    Some((
        match target {
            Target::At(a) => a,
            Target::Len(_) => LOCKED.load(Relaxed),
            Target::All => 0,
        },
        mode,
    ))
}

/// What is to be applied: dword index -> (value, mask); and, for the check,
/// each run (offset in the copy, its bytes).
static PATCHES: Mutex<BTreeMap<u64, BTreeMap<u32, (u32, u32)>>> = Mutex::new(BTreeMap::new());
static RUNS: Mutex<Vec<(u32, Vec<u8>)>> = Mutex::new(Vec::new());
/// `all:on`: per copy, the records of each page merged, in the order made
/// (sorted and joined at [`take`]; a later record's bytes win over an
/// earlier one's). Each page's records are in dword order, `lo..=hi`.
struct Chunk {
    lo: u32,
    hi: u32,
    recs: Vec<[u32; 3]>,
}
static FAST: Mutex<Vec<(u64, Vec<Chunk>)>> = Mutex::new(Vec::new());

/// Records that went to the card side, per copy and send (`crate::host`'s
/// order of the queue), in dword order: the bytes they cover hold the
/// title's, newer than what a fetch made before that send read from the
/// card ([`skip`]), as the copy's `sent` ranges do for its copy commands.
/// Kept until a fetch sent after them has been merged ([`forget_sent`]).
static SENT: Mutex<Vec<(u64, u64, std::sync::Arc<Vec<[u32; 3]>>)>> = Mutex::new(Vec::new());

fn selfcheck() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_BATCH_SELFCHECK").is_some_and(|v| v == "1"))
}
/// Self-check totals: pages, bytes that differ, record bytes outside their
/// page, bytes looked at.
static SELF: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether every copy goes through the batch (`all:on`): the fast path.
pub(crate) fn all_on() -> bool {
    matches!(flag(), Some((Target::All, Mode::On))) && ready()
}

/// A dispatch could not be recorded: no more records are made, the copy
/// commands are used from here on.
static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the pass exists (compiled once the host device does) and no
/// dispatch has failed: records are made only then, so that a record never
/// waits for a pass that cannot run.
fn ready() -> bool {
    !FAILED.load(Relaxed) && crate::host::host().is_some_and(|h| pass(h.device()).is_some())
}

/// `all:on`: the title's bytes of one page of the copy at `copy`
/// (`title`, at offset `at` of the copy) against what the copy's CPU side
/// holds there (`held`): every byte that differs becomes part of a record.
/// Eight bytes are compared at a time; only those that differ are looked at
/// one by one. Returns the bytes that differ.
pub(crate) fn record_page(copy: u64, at: usize, title: &[u8], held: &[u8]) -> usize {
    let n = title.len().min(held.len());
    let mut out: Vec<[u32; 3]> = Vec::new();
    let mut changed = 0usize;
    let mut i = 0;
    while i < n {
        let w = (n - i).min(8);
        if title[i..i + w] == held[i..i + w] {
            i += w;
            continue;
        }
        for k in i..i + w {
            if title[k] != held[k] {
                changed += 1;
                let o = at + k;
                let (dword, shift) = ((o / 4) as u32, (o % 4) * 8);
                match out.last_mut() {
                    Some(r) if r[0] == dword => {
                        r[1] = (r[1] & !(0xFF << shift)) | (u32::from(title[k]) << shift);
                        r[2] |= 0xFF << shift;
                    }
                    _ => out.push([dword, u32::from(title[k]) << shift, 0xFF << shift]),
                }
            }
        }
        i += w;
    }
    // `VICTORY_MERGE_BATCH_SELFCHECK=1`: the page's records applied to a base
    // that differs from the CPU side in every byte must give that base with
    // exactly the title's changed bytes put in (the copy commands' result),
    // and nothing else.
    if selfcheck() {
        let base: Vec<u8> = held[..n].iter().map(|b| b ^ 0x5A).collect();
        let mut got = base.clone();
        for r in &out {
            for j in 0..4usize {
                if r[2] & (0xFF << (j * 8)) != 0 {
                    let off = r[0] as usize * 4 + j;
                    if off >= at && off < at + n {
                        got[off - at] = (r[1] >> (j * 8)) as u8;
                    } else {
                        SELF[2].fetch_add(1, Relaxed);
                    }
                }
            }
        }
        let differ = (0..n).filter(|&k| got[k] != if title[k] != held[k] { title[k] } else { base[k] }).count();
        SELF[0].fetch_add(1, Relaxed);
        SELF[1].fetch_add(differ as u64, Relaxed);
        SELF[3].fetch_add(n as u64, Relaxed);
    }
    if !out.is_empty() {
        STATS[2].fetch_add(changed as u64, Relaxed);
        NOW.fetch_add(out.len() as u64, Relaxed);
        STATS[1].fetch_add(out.len() as u64, Relaxed);
        let chunk = Chunk { lo: out[0][0], hi: out[out.len() - 1][0], recs: out };
        let mut f = FAST.lock().unwrap_or_else(|e| e.into_inner());
        match f.iter_mut().find(|(c, _)| *c == copy) {
            Some((_, v)) => v.push(chunk),
            None => f.push((copy, vec![chunk])),
        }
    }
    changed
}

/// Counts since the last line: submissions with runs of the copy, runs,
/// run bytes, runs past the cap, most runs in one submission, dword records,
/// dispatches; and the runs of the submission being prepared.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
static NOW: AtomicU64 = AtomicU64::new(0);
/// The check's totals: compared submissions, bytes compared, bytes that
/// differ from the copy commands' result, bytes changed outside the runs,
/// run bytes.
static CHECKED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// A merge of the named copy found `runs` changed runs; `capped`: it stops
/// there and the flights are settled (the current path).
pub(crate) fn counted(runs: usize, capped: bool) {
    NOW.fetch_add(runs as u64, Relaxed);
    STATS[1].fetch_add(runs as u64, Relaxed);
    if capped {
        STATS[3].fetch_add(runs as u64, Relaxed);
    }
}

/// One run that goes into the copy at `offset` (bytes of the copy).
pub(crate) fn record(copy: u64, offset: usize, bytes: &[u8], check: bool) {
    STATS[2].fetch_add(bytes.len() as u64, Relaxed);
    let mut all = PATCHES.lock().unwrap_or_else(|e| e.into_inner());
    let p = all.entry(copy).or_default();
    for (k, &b) in bytes.iter().enumerate() {
        let at = offset + k;
        let (dword, shift) = ((at / 4) as u32, (at % 4) * 8);
        let e = p.entry(dword).or_insert((0, 0));
        e.0 = (e.0 & !(0xFF << shift)) | (u32::from(b) << shift);
        e.1 |= 0xFF << shift;
    }
    if check {
        RUNS.lock().unwrap_or_else(|e| e.into_inner()).push((offset as u32, bytes.to_vec()));
    }
}

/// The records of this submission per copy, and the runs for the check; the
/// per-submission count is closed. `sent_as`: with `register` (the records
/// go to the card side in that send, and the relay lands flights), they
/// become [`SENT`] in the same step as they stop being pending, so that no
/// landing finds them in neither.
pub(crate) fn take(sent_as: u64, register: bool) -> (Vec<(u64, std::sync::Arc<Vec<[u32; 3]>>)>, Vec<(u32, Vec<u8>)>) {
    let n = NOW.swap(0, Relaxed);
    if n > 0 {
        STATS[0].fetch_add(1, Relaxed);
        STATS[4].fetch_max(n, Relaxed);
    }
    let mut fast_lock = FAST.lock().unwrap_or_else(|e| e.into_inner());
    let fast = std::mem::take(&mut *fast_lock);
    let mut joined: Vec<(u64, Vec<[u32; 3]>)> = Vec::new();
    for (copy, chunks) in fast {
        let mut v: Vec<[u32; 3]> = chunks.into_iter().flat_map(|c| c.recs).collect();
        // In dword order, the order made kept among equal dwords; then one
        // record a dword, the later bytes over the earlier.
        v.sort_by_key(|r| r[0]);
        let mut one: Vec<[u32; 3]> = Vec::with_capacity(v.len());
        for r in v {
            match one.last_mut() {
                Some(l) if l[0] == r[0] => {
                    l[1] = (l[1] & !r[2]) | (r[1] & r[2]);
                    l[2] |= r[2];
                }
                _ => one.push(r),
            }
        }
        joined.push((copy, one));
    }
    let all = std::mem::take(&mut *PATCHES.lock().unwrap_or_else(|e| e.into_inner()));
    let out: Vec<(u64, Vec<[u32; 3]>)> = joined.into_iter().chain(all
        .into_iter()
        .map(|(copy, p)| (copy, p.into_iter().map(|(d, (v, m))| [d, v, m]).collect::<Vec<_>>())))
        .filter(|(_, r)| !r.is_empty())
        .collect();
    STATS[5].fetch_add(out.iter().map(|(_, r)| r.len() as u64).sum(), Relaxed);
    let out: Vec<(u64, std::sync::Arc<Vec<[u32; 3]>>)> = out.into_iter().map(|(c, r)| (c, std::sync::Arc::new(r))).collect();
    if register {
        let mut sent = SENT.lock().unwrap_or_else(|e| e.into_inner());
        sent.extend(out.iter().map(|(c, r)| (*c, sent_as, r.clone())));
    }
    drop(fast_lock);
    let runs = std::mem::take(&mut *RUNS.lock().unwrap_or_else(|e| e.into_inner()));
    (out, runs)
}

/// The copy at `copy`: its records sent in sends up to `fetched` are no
/// longer needed by a landing (as `Shadow`'s `sent` ranges are retained).
pub(crate) fn forget_sent(copy: u64, fetched: u64) {
    let mut sent = SENT.lock().unwrap_or_else(|e| e.into_inner());
    if !sent.is_empty() {
        sent.retain(|s| s.0 != copy || s.1 > fetched);
    }
}

/// Byte ranges of offsets `from..to` of the copy at `copy` that a landing of
/// a fetch made before send `fetched_after` must leave alone: bytes of
/// records not sent yet, and of records sent after it. Exactly the bytes the
/// title changed, as the copy's `dirty` and `sent` ranges are for its copy
/// commands; added to `out`.
pub(crate) fn skip(copy: u64, from: usize, to: usize, fetched_after: u64, out: &mut Vec<(usize, usize)>) {
    if from >= to || !matches!(flag(), Some((_, Mode::On))) {
        return;
    }
    let (dlo, dhi) = ((from / 4) as u32, ((to - 1) / 4) as u32);
    let mut add = |r: &[u32; 3]| {
        for j in 0..4usize {
            if r[2] & (0xFF << (j * 8)) != 0 {
                let off = r[0] as usize * 4 + j;
                if off >= from && off < to {
                    match out.last_mut() {
                        Some(last) if last.1 == off => last.1 = off + 1,
                        _ => out.push((off, off + 1)),
                    }
                }
            }
        }
    };
    let in_range = |recs: &[[u32; 3]]| {
        let a = recs.partition_point(|r| r[0] < dlo);
        let b = recs.partition_point(|r| r[0] <= dhi);
        (a, b)
    };
    {
        let fast = FAST.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, chunks)) = fast.iter().find(|(c, _)| *c == copy) {
            for c in chunks.iter().filter(|c| c.hi >= dlo && c.lo <= dhi) {
                let (a, b) = in_range(&c.recs);
                c.recs[a..b].iter().for_each(&mut add);
            }
        }
        let sent = SENT.lock().unwrap_or_else(|e| e.into_inner());
        for (_, _, recs) in sent.iter().filter(|s| s.0 == copy && s.1 > fetched_after) {
            let (a, b) = in_range(recs);
            recs[a..b].iter().for_each(&mut add);
        }
    }
    let patches = PATCHES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = patches.get(&copy) {
        for (&d, &(v, m)) in p.range(dlo..=dhi) {
            add(&[d, v, m]);
        }
    }
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0)"
cbuffer C : register(b0) { uint count; };
ByteAddressBuffer records : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= count) return;
    uint3 r = records.Load3(id.x * 12);
    uint old = dst.Load(r.x * 4);
    dst.Store(r.x * 4, (old & ~r.z) | (r.y & r.z));
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn pass(device: &ID3D12Device) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; out pointers ours.
        let compiled = unsafe {
            D3DCompile(SOURCE.as_ptr().cast(), SOURCE.len(), s!("merge_batch"), None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
        };
        if compiled.is_err() {
            if let Some(errors) = errors {
                // SAFETY: the blob's own bytes.
                let text = unsafe { std::slice::from_raw_parts(errors.GetBufferPointer().cast::<u8>(), errors.GetBufferSize()) };
                tracing::warn!("merge batch pass: {}", String::from_utf8_lossy(text));
            }
            return None;
        }
        let code = code?;
        // SAFETY: the blob's own bytes; the root signature is in the shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).ok()?;
            let pso = device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() },
                    ..Default::default()
                })
                .ok()?;
            Some(Pass { root, pso })
        }
    })
    .as_ref()
}

/// Buffers whose last use ends when the host's fence reaches `until`.
struct Held {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
    until: u64,
}
// SAFETY: host objects (free-threaded) and a mapping only used under the lock.
unsafe impl Send for Held {}

static UPLOADS: Mutex<Vec<Held>> = Mutex::new(Vec::new());
static TWINS: Mutex<Vec<Held>> = Mutex::new(Vec::new());
/// Check read-backs not compared yet: buffer (before, then twin), size, runs.
static PENDING: Mutex<Vec<(Held, Vec<(u32, Vec<u8>)>)>> = Mutex::new(Vec::new());

fn reuse(pool: &Mutex<Vec<Held>>, size: u64, completed: u64, until: u64, make: impl FnOnce(u64) -> Option<(ID3D12Resource, usize)>) -> Option<(ID3D12Resource, usize)> {
    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = pool.iter_mut().find(|h| h.until <= completed && h.size >= size) {
        h.until = until;
        return Some((h.buffer.clone(), h.data));
    }
    let size = size.next_power_of_two().max(1 << 16);
    let (buffer, data) = make(size)?;
    pool.push(Held { buffer: buffer.clone(), data, size, until });
    Some((buffer, data))
}

/// Records the dispatch of `records` into `list`, onto `dst` (in state
/// UNORDERED_ACCESS). The records' upload memory is kept until the host's
/// fence reaches `until`.
pub(crate) fn dispatch(host: &crate::host::Host, list: &ID3D12GraphicsCommandList, dst: &ID3D12Resource, records: &[[u32; 3]], completed: u64, until: u64) -> bool {
    if records.is_empty() {
        return true;
    }
    let failed = || {
        if !FAILED.swap(true, Relaxed) {
            tracing::error!(
                records = records.len(),
                "VICTORY_MERGE_BATCH: a dispatch could not be recorded (no pass or no upload memory); the copy commands are used from now on, and this submission's batched bytes of the copy reach the card only when the title writes them again"
            );
        }
        false
    };
    let Some(p) = pass(&host.device()) else {
        return failed();
    };
    let bytes = (records.len() * 12) as u64;
    let Some((upload, data)) = reuse(&UPLOADS, bytes, completed, until, |n| host.shared_buffer(n).map(|(b, d)| (b, d as usize))) else {
        return failed();
    };
    // SAFETY: the mapping holds `bytes` and more; nothing reads it until the
    // list runs.
    unsafe {
        std::ptr::copy_nonoverlapping(records.as_ptr().cast::<u8>(), data as *mut u8, bytes as usize);
        list.SetComputeRootSignature(&p.root);
        list.SetPipelineState(&p.pso);
        list.SetComputeRoot32BitConstant(0, records.len() as u32, 0);
        list.SetComputeRootShaderResourceView(1, upload.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(2, dst.GetGPUVirtualAddress());
        list.Dispatch((records.len() as u32).div_ceil(64), 1, 1);
    }
    STATS[6].fetch_add(1, Relaxed);
    true
}

/// Check mode: a twin of `card` (`size` bytes, the copy's card side) and a
/// read-back of before and after; records into `list`, which must find
/// `card` in the common state and leaves it in COPY_DEST (`ranges`: the
/// copy commands follow) or COMMON.
pub(crate) fn check(
    host: &crate::host::Host,
    list: &ID3D12GraphicsCommandList,
    card: &ID3D12Resource,
    size: u64,
    records: &[[u32; 3]],
    runs: Vec<(u32, Vec<u8>)>,
    ranges: bool,
    completed: u64,
    until: u64,
) {
    use crate::host::transition;
    let Some((twin, _)) = reuse(&TWINS, size, completed, until, |n| host.uav_buffer(n).map(|b| (b, 0))) else {
        return;
    };
    let Some((back, data)) = host.shared_buffer(2 * size) else {
        return;
    };
    // SAFETY: an open list; live buffers of at least `size` bytes (twin) and
    // `2 * size` (back); `card` in the common state.
    unsafe {
        list.ResourceBarrier(&[transition(card, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&back, 0, card, 0, size);
        list.CopyBufferRegion(&twin, 0, card, 0, size);
        list.ResourceBarrier(&[
            transition(card, D3D12_RESOURCE_STATE_COPY_SOURCE, if ranges { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_COMMON }),
            transition(&twin, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
    }
    dispatch(host, list, &twin, records, completed, until);
    // SAFETY: as above.
    unsafe {
        list.ResourceBarrier(&[transition(&twin, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyBufferRegion(&back, size, &twin, 0, size);
        list.ResourceBarrier(&[transition(&twin, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
    }
    PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((Held { buffer: back, data: data as usize, size, until }, runs));
}

/// Compares the read-backs whose lists have run (the host's fence at
/// `completed`).
pub(crate) fn compare_done(completed: u64) {
    let done: Vec<(Held, Vec<(u32, Vec<u8>)>)> = {
        let mut p = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        let (done, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut *p).into_iter().partition(|(h, _)| h.until <= completed);
        *p = keep;
        done
    };
    for (h, runs) in done {
        let n = h.size as usize;
        // SAFETY: the read-back's mapping holds the card side before (first
        // `size` bytes) and the twin after; the card has written both.
        let (before, twin) = unsafe { (std::slice::from_raw_parts(h.data as *const u8, n), std::slice::from_raw_parts((h.data + n) as *const u8, n)) };
        // What the copy commands put there: each run's bytes over before.
        let mut expect = before.to_vec();
        let mut in_run = vec![false; n];
        let mut run_bytes = 0u64;
        for (at, bytes) in &runs {
            let at = *at as usize;
            if at + bytes.len() > n {
                continue;
            }
            expect[at..at + bytes.len()].copy_from_slice(bytes);
            in_run[at..at + bytes.len()].iter_mut().for_each(|b| *b = true);
            run_bytes += bytes.len() as u64;
        }
        let differ = expect.iter().zip(twin).filter(|(a, b)| a != b).count() as u64;
        let outside = (0..n).filter(|&i| !in_run[i] && twin[i] != before[i]).count() as u64;
        CHECKED[0].fetch_add(1, Relaxed);
        CHECKED[1].fetch_add(n as u64, Relaxed);
        CHECKED[2].fetch_add(differ, Relaxed);
        CHECKED[3].fetch_add(outside, Relaxed);
        CHECKED[4].fetch_add(run_bytes, Relaxed);
        if differ > 0 && CHECKED[0].load(Relaxed) < 50 {
            let first = (0..n).find(|&i| expect[i] != twin[i]).unwrap_or(0);
            tracing::warn!(first = format_args!("{first:#x}"), differ, outside, "merge batch check: the dispatch's result differs from the copy commands'");
        }
    }
}

/// With the frame-rate line: the counts since the last one.
pub(crate) fn line(frames: f64) {
    let Some((at, mode)) = named() else {
        return;
    };
    let s: Vec<u64> = STATS.iter().map(|a| a.swap(0, Relaxed)).collect();
    let c: Vec<u64> = CHECKED.iter().map(|a| a.load(Relaxed)).collect();
    if selfcheck() {
        tracing::info!(
            "merge batch self-check so far: {} pages, {} bytes looked at, {} bytes differ from the copy commands' result, {} record bytes outside their page",
            SELF[0].load(Relaxed),
            SELF[3].load(Relaxed),
            SELF[1].load(Relaxed),
            SELF[2].load(Relaxed)
        );
    }
    tracing::info!(
        "merge batch (VICTORY_MERGE_BATCH, copy {at:#x} (0: every copy), {}): per frame {:.1} submissions with its runs, {:.0} changed runs ({:.0} bytes) \
         = copy commands the current merge needs, {:.0} of them past the cap; most in one submission {}; {:.0} dword records, {:.1} dispatches; \
         check so far: {} submissions compared, {} of {} bytes differ from the copy commands' result, {} bytes changed outside the runs ({} run bytes)",
        match mode {
            Mode::Count => "count",
            Mode::Check => "check",
            Mode::On => "on",
        },
        s[0] as f64 / frames,
        s[1] as f64 / frames,
        s[2] as f64 / frames,
        s[3] as f64 / frames,
        s[4],
        s[5] as f64 / frames,
        s[6] as f64 / frames,
        c[0],
        c[2],
        c[1],
        c[3],
        c[4],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests share the module's statics ([`take`] drains every copy's
    /// records): one at a time.
    static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

    /// A landing's skip ranges are exactly the bytes of pending records and
    /// of records sent after its fetch, nothing around them.
    #[test]
    fn landings_skip_exactly_the_batched_bytes() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        // (`skip` asks for mode on; the flag is read once per process.)
        unsafe { std::env::set_var("VICTORY_MERGE_BATCH", "all:on") };
        assert!(matches!(flag(), Some((_, Mode::On))), "the flag must be on for this test");
        let copy = 0x5150;
        // Pending: bytes 9 and 11 of dword 2 (not 10), byte 0 of dword 5.
        FAST.lock().unwrap().push((copy, vec![Chunk { lo: 2, hi: 5, recs: vec![[2, 0, 0xFF00_FF00], [5, 0, 0xFF]] }]));
        // Sent in send 4 (bytes 32..34) and send 9 (byte 40).
        SENT.lock().unwrap().push((copy, 4, std::sync::Arc::new(vec![[8, 0, 0xFFFF]])));
        SENT.lock().unwrap().push((copy, 9, std::sync::Arc::new(vec![[10, 0, 0xFF]])));
        let ranges = |after: u64| {
            let mut out = Vec::new();
            skip(copy, 0, 64, after, &mut out);
            out.sort_unstable();
            out
        };
        assert_eq!(ranges(0), vec![(9, 10), (11, 12), (20, 21), (32, 34), (40, 41)]);
        assert_eq!(ranges(4), vec![(9, 10), (11, 12), (20, 21), (40, 41)], "send 4 is not after a fetch made before send 4");
        let mut cut = Vec::new();
        skip(copy, 10, 33, 0, &mut cut);
        cut.sort_unstable();
        assert_eq!(cut, vec![(11, 12), (20, 21), (32, 33)], "only the window asked for");
        forget_sent(copy, 9);
        assert_eq!(ranges(0), vec![(9, 10), (11, 12), (20, 21)]);
        FAST.lock().unwrap().retain(|(c, _)| *c != copy);
    }

    /// The records of pages applied to a base that holds other bytes (as a
    /// card side the shaders wrote does) give that base with exactly the
    /// bytes the title changed put in, at any alignment of the page in the
    /// copy, and records of two pages that share a dword join.
    #[test]
    fn records_put_in_exactly_the_changed_bytes() {
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..3000u32 {
            let n = 1 + (next() % 300) as usize;
            let at = (next() % 9) as usize;
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            for _ in 0..(next() % 20) {
                let i = (next() as usize) % n;
                let len = 1 + (next() as usize) % 6;
                for k in i..(i + len).min(n) {
                    title[k] = title[k].wrapping_add(1 + (next() % 255) as u8);
                }
            }
            // As two pages when long enough: the cut may fall inside a dword.
            let cut = if n > 8 { 1 + (next() as usize) % (n - 1) } else { n };
            let changed = record_page(7, at, &title[..cut], &held[..cut]) + record_page(7, at + cut, &title[cut..], &held[cut..]);
            let (all, _) = take(0, false);
            let records: Vec<[u32; 3]> = all.into_iter().filter(|(c, _)| *c == 7).flat_map(|(_, r)| r.iter().copied().collect::<Vec<_>>()).collect();
            assert!(records.windows(2).all(|w| w[0][0] < w[1][0]), "case {case}: one record a dword, in order");
            let base: Vec<u8> = held.iter().map(|b| b ^ 0x5A).collect();
            let mut got = base.clone();
            for r in &records {
                for j in 0..4usize {
                    if r[2] & (0xFF << (j * 8)) != 0 {
                        let off = r[0] as usize * 4 + j;
                        assert!(off >= at && off < at + n, "case {case}: a record byte outside the pages");
                        got[off - at] = (r[1] >> (j * 8)) as u8;
                    }
                }
            }
            let expect: Vec<u8> = (0..n).map(|k| if title[k] != held[k] { title[k] } else { base[k] }).collect();
            assert_eq!(got, expect, "case {case}");
            assert_eq!(changed, (0..n).filter(|&k| title[k] != held[k]).count(), "case {case}");
        }
    }
}
