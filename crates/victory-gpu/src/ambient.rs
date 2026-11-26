//! Indirect lighting when screen probe rays are disabled.
//! Lighting math adapted from Belobug/VIctory-Belobug-AMD at 4f62c2a.

use std::sync::{Arc, Mutex, Once, OnceLock};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host::{HostList, host};
use crate::object::This;
use crate::resource::HostTexture;

pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
pub(crate) const SKY: &[u8] = b"ps_sky_paraboloid_reflection";
pub(crate) const REFLECTED: &[u8] = b"CS_UpdateHistory";
pub(crate) const LIGHT: (u8, u32) = (1, 176);
pub(crate) const DEPTH: (u8, u32) = (0, 464);
pub(crate) const REFLECTIONS: (u8, u32) = (1, 40);

fn level() -> Option<f32> {
    static LEVEL: OnceLock<Option<f32>> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = std::env::var("VICTORY_AMBIENT")
            .ok()?
            .trim()
            .parse::<f32>()
            .ok()?;
        (level.is_finite() && (0.0..=65408.0).contains(&level)).then_some(level)
    })
}

pub(crate) fn enabled() -> bool {
    level().is_some()
}

static SKY_IMAGE: Mutex<Option<Arc<HostTexture>>> = Mutex::new(None);

pub(crate) fn sky_drawn(target: This) {
    let Some(copy) = crate::resource::host_copy(target) else {
        return;
    };
    // SAFETY: the host copy keeps the resource alive.
    let desc = unsafe { copy.texture.GetDesc() };
    if desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT
        && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
        && desc.Width == 128
        && desc.Height == 128
    {
        *SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some(copy);
    }
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=13, b0), DescriptorTable(UAV(u0), SRV(t0), SRV(t1))"
cbuffer Floor : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
    float gain; float least;
    float held; float floor_part; float power;
};
Texture2DArray<float4> sky : register(t1);
// The light, from the sky's mean when it follows the sky (gain above 0).
float3 from_around() {
    if (gain <= 0.0) return tint * level;
    uint sw, sh, slices;
    sky.GetDimensions(sw, sh, slices);
    float3 sum = 0.0;
    const int N = 8;
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            // The sun's own texels are held down: it lights by itself.
            float3 there = sky.Load(int4((x * 2 + 1) * sw / (N * 2), (y * 2 + 1) * sh / (N * 2), 0, 0)).rgb;
            sum += clamp(there, 0.0, 8.0);
        }
    float3 mean = sum / (N * N);
    float m = dot(mean, float3(0.2126, 0.7152, 0.0722));
    // The sky's colour, half washed out (the ground lights too); a dark
    // sky has none worth following.
    const float3 NIGHT = float3(0.6, 0.75, 1.0);
    float3 hue = lerp(NIGHT, lerp(float3(1.0, 1.0, 1.0), mean / max(m, 1e-6), 0.5), saturate(m / 0.05));
    return hue * (gain * pow(max(m, 0.0), power) + least);
}
RWTexture2D<uint> light : register(u0);
Texture2D<float> depth : register(t0);
// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}
float3 unpack(uint v) {
    return float3(v & 511, (v >> 9) & 511, (v >> 18) & 511) * exp2(float(v >> 27) - 24.0);
}
uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
groupshared float3 ambient;
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID, uint lane : SV_GroupIndex) {
    if (lane == 0) ambient = from_around();
    GroupMemoryBarrierWithGroupSync();
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    // What the resolve left is not kept: without rays its inputs are never
    // written and it holds whatever their memory did (bright magenta on
    // translucent surfaces, blocks of colours indoors: run v6, 2026-11-25).
    if (held > 0.0) {
        // Reflections: no brighter than a part of the light around.
        const float3 LUMA = float3(0.2126, 0.7152, 0.0722);
        float3 seen_there = unpack(light[at.xy]);
        float most = dot(ambient, LUMA) * held;
        float has = dot(seen_there, LUMA);
        float3 floor_light = ambient * floor_part;
        if (has > most) light[at.xy] = pack(seen_there * (most / has));
        else if (has < dot(floor_light, LUMA)) light[at.xy] = pack(floor_light);
        return;
    }
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(w, h)) : 1.0;
    light[at.xy] = pack(ambient * open);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the compiler reads the static source and fills the two local output pointers.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("ambient"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(error) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the compiler's blob owns this buffer.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!(details = %String::from_utf8_lossy(bytes), "ambient shader failed to compile");
        }
        return Err(error);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob holds both the shader and its embedded root signature.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root = device.CreateRootSignature(0, bytes)?;
        let pipeline = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        });
        // The call borrows the root signature, which the pass continues to own.
        Ok(Pass {
            root,
            pipeline: pipeline?,
        })
    }
}

pub(crate) fn record(list: &HostList, image: This, depth: Option<This>, reflection: bool) -> bool {
    let (Some(level), Some(host), Some(image_copy)) =
        (level(), host(), crate::resource::host_copy(image))
    else {
        return false;
    };
    let texture = &image_copy.texture;
    // SAFETY: the host copy owns the texture.
    let desc = unsafe { texture.GetDesc() };
    if desc.Format != DXGI_FORMAT_R32_UINT
        || desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || desc.DepthOrArraySize != 1
        || (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS).0 == 0
    {
        static UNEXPECTED: Once = Once::new();
        UNEXPECTED.call_once(|| {
            tracing::warn!(
                format = desc.Format.0,
                "ambient light target has an unexpected descriptor"
            )
        });
        return false;
    }

    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(
                    |error| tracing::warn!(%error, "could not create ambient lighting pipeline"),
                )
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    let sky = SKY_IMAGE.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let depth = depth.and_then(crate::resource::host_copy).filter(|copy| {
        // SAFETY: the host copy owns the texture.
        let d = unsafe { copy.texture.GetDesc() };
        d.Format == DXGI_FORMAT_R32G8X24_TYPELESS
            && (d.Width, d.Height) == (desc.Width, desc.Height)
    });

    // Descriptors remain immutable until this list retires.
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // SAFETY: a live device and a valid descriptor heap description.
    let heap: ID3D12DescriptorHeap = match unsafe {
        host.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
    } {
        Ok(heap) => heap,
        Err(error) => {
            static FAILED: Once = Once::new();
            FAILED.call_once(
                || tracing::warn!(%error, "could not allocate ambient lighting descriptors"),
            );
            return false;
        }
    };

    list.keep(&heap);
    crate::list::ensure_state(list, image, D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32);
    if let Some(sky) = &sky {
        list.settle(
            sky,
            (D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)
                .0 as u32,
        );
    }
    if let Some(depth) = &depth {
        list.keep(&depth.texture);
    }
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open graphics/compute list and resources held through execution.
    // The resolve has already bound its depth as a shader resource.
    unsafe {
        let base = heap.GetCPUDescriptorHandleForHeapStart();
        let step = host.device.GetDescriptorHandleIncrementSize(kind) as usize;
        let handle = |slot| D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: base.ptr + slot * step,
        };
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device.CreateUnorderedAccessView(
            texture,
            None::<&ID3D12Resource>,
            Some(&uav),
            handle(0),
        );
        let depth_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        host.device.CreateShaderResourceView(
            depth.as_ref().map(|copy| &copy.texture),
            Some(&depth_srv),
            handle(1),
        );
        let sky_srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                    MipLevels: 1,
                    ArraySize: 1,
                    ..Default::default()
                },
            },
        };
        host.device.CreateShaderResourceView(
            sky.as_ref().map(|copy| &copy.texture),
            Some(&sky_srv),
            handle(2),
        );
        list.list.ResourceBarrier(std::slice::from_ref(&barrier));
        let gpu = heap.GetGPUDescriptorHandleForHeapStart();
        list.list.SetDescriptorHeaps(&[Some(heap)]);
        list.list.SetComputeRootSignature(&pass.root);
        list.list.SetPipelineState(&pass.pipeline);
        let strength = if depth.is_some() { 1.0 } else { 0.0 };
        let gain = if sky.is_some() { 0.36 } else { 0.0 };
        let held = if reflection { 0.5 } else { 0.0 };
        let constants: [f32; 13] = [
            level, 1.0, 0.98, 0.94, strength, 0.6, 0.1, 0.45, gain, 0.05, held, 0.0, 0.4,
        ];
        list.list.SetComputeRoot32BitConstants(
            0,
            constants.len() as u32,
            constants.as_ptr().cast(),
            0,
        );
        list.list.SetComputeRootDescriptorTable(1, gpu);
        list.list
            .Dispatch((desc.Width as u32).div_ceil(8), desc.Height.div_ceil(8), 1);
        list.list.ResourceBarrier(std::slice::from_ref(&barrier));
    }
    static DIFFUSE: Once = Once::new();
    static REFLECTION: Once = Once::new();
    static SKY_ACTIVE: Once = Once::new();
    if reflection {
        REFLECTION.call_once(|| tracing::info!("ambient lighting limited glossy reflections"));
    } else {
        DIFFUSE.call_once(|| {
            tracing::info!(
                level,
                width = desc.Width,
                height = desc.Height,
                depth = depth.is_some(),
                "ambient light applied to screen probes"
            )
        });
    }
    if sky.is_some() {
        SKY_ACTIVE.call_once(|| tracing::info!("ambient lighting follows the sky dome"));
    }
    true
}
