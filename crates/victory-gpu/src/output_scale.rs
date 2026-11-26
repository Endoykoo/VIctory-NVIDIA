//! Scales a finished title frame into a smaller presentation buffer.

use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{Result, s};

use crate::host::{Host, HostList};

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0)), RootConstants(num32BitConstants=2, b0), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> frame : register(t0);
SamplerState linearClamp : register(s0);
cbuffer OutputSize : register(b0) { float2 inverseSize; }

float4 corner(uint index : SV_VertexID) : SV_Position {
    float2 uv = float2((index << 1) & 2, index & 2);
    return float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
}

[RootSignature(RS)]
float4 sampleFrame(float4 position : SV_Position) : SV_Target {
    return frame.SampleLevel(linearClamp, position.xy * inverseSize, 0);
}
"#;

/// Preserve aspect ratio and leave already-small frames at their original size.
pub(crate) fn dimensions(width: u32, height: u32) -> (u32, u32) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| std::env::var_os("VICTORY_OUTPUT_720P").is_some_and(|v| v == "1"));
    if !enabled || (width <= 1280 && height <= 720) || width == 0 || height == 0 {
        return (width, height);
    }
    let (w, h) = (u64::from(width), u64::from(height));
    if w * 720 > h * 1280 {
        (1280, (h * 1280 / w).max(1) as u32)
    } else {
        ((w * 720 / h).max(1) as u32, 720)
    }
}

#[derive(Clone)]
pub(crate) struct Pass {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}

impl Pass {
    pub(crate) fn get(host: &Host, format: DXGI_FORMAT) -> Option<Self> {
        static PASSES: Mutex<Vec<(DXGI_FORMAT, Option<Pass>)>> = Mutex::new(Vec::new());
        let mut passes = PASSES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pass)) = passes.iter().find(|(key, _)| *key == format) {
            return pass.clone();
        }
        let pass = match Self::make(&host.device, format) {
            Ok(pass) => Some(pass),
            Err(error) => {
                tracing::warn!(%error, format = format.0, "could not create the 720p output scaler");
                None
            }
        };
        passes.push((format, pass.clone()));
        pass
    }

    fn make(device: &ID3D12Device, format: DXGI_FORMAT) -> Result<Self> {
        let vs = crate::host::compile(SOURCE, s!("corner"), s!("vs_5_1"))?;
        let ps = crate::host::compile(SOURCE, s!("sampleFrame"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
        formats[0] = format;
        let blend = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let stencil = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        // SAFETY: shader data and the borrowed root signature outlive pipeline creation.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps)?;
            let pipeline =
                device.CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: code(&vs),
                    PS: code(&ps),
                    BlendState: D3D12_BLEND_DESC {
                        RenderTarget: [blend; 8],
                        ..Default::default()
                    },
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: true.into(),
                        ..Default::default()
                    },
                    DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                        DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                        FrontFace: stencil,
                        BackFace: stencil,
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                })?;
            Ok(Self { root, pipeline })
        }
    }

    pub(crate) fn descriptors(host: &Host) -> Result<ID3D12DescriptorHeap> {
        // SAFETY: plain heap creation on a live device.
        unsafe {
            host.device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
        }
    }

    pub(crate) fn draw(
        &self,
        host: &Host,
        commands: &HostList,
        descriptors: &ID3D12DescriptorHeap,
        source: &ID3D12Resource,
        target: &ID3D12Resource,
        (width, height, format): (u32, u32, DXGI_FORMAT),
    ) {
        commands.keep(descriptors);
        commands.keep(source);
        let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let rtv = host.rtv(Some(target), None);
        let list = &commands.list;
        let inverse_size = [1.0 / width as f32, 1.0 / height as f32];
        // SAFETY: the caller has transitioned source and target for sampling and rendering.
        unsafe {
            host.device.CreateShaderResourceView(
                source,
                Some(&view),
                descriptors.GetCPUDescriptorHandleForHeapStart(),
            );
            list.SetDescriptorHeaps(&[Some(descriptors.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.SetPipelineState(&self.pipeline);
            list.SetGraphicsRootDescriptorTable(
                0,
                descriptors.GetGPUDescriptorHandleForHeapStart(),
            );
            list.SetGraphicsRoot32BitConstants(1, 2, inverse_size.as_ptr().cast(), 0);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: width as f32,
                Height: height as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
        }
    }
}
