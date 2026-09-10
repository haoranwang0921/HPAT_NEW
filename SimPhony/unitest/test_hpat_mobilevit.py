"""
HPAT + MobileViT simulation: map MobileViT XXS layers to HPAT PDPU architecture
and run the full SimPhony energy/area pipeline.

Uses raw PyTorch layers because mmcv/mmengine are unavailable on Windows.
MatMul layers in self-attention are wrapped in MatMul for detection.
"""

import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch import Tensor

from onnarchsim.simulator import ONNArchSimulator


# ---------------------------------------------------------------------------
# Minimal MatMul wrapper so extract_layer_info can detect attention MatMul ops
# ---------------------------------------------------------------------------
class MatMul(nn.Module):
    """Wraps torch.matmul as a named module so SimPhony can map it."""

    def __init__(self):
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return torch.matmul(a, b)


# ---------------------------------------------------------------------------
# MobileViT building blocks (raw PyTorch, no mmcv dependency)
# ---------------------------------------------------------------------------
class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual: expand -> depthwise -> project."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = int(inp * expand_ratio)
        self.use_residual = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class Attention(nn.Module):
    """Multi-head self-attention with MatMul for Q*K^T and attn@V."""

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.proj = nn.Linear(self.inner_dim, dim, bias=False)
        self.matmul_qk = MatMul()  # Q @ K^T
        self.matmul_av = MatMul()  # attn @ V

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, d]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.matmul_qk(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = self.matmul_av(attn, v)
        out = out.transpose(1, 2).contiguous().reshape(B, N, self.inner_dim)
        return self.proj(out)


class TransformerLayer(nn.Module):
    """Pre-LN transformer: attention + MLP with residuals."""

    def __init__(self, dim: int, num_heads: int = 4, dim_head: int = 8,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dim_head)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    """MobileViT fusion block: local conv -> global transformer -> fusion conv."""

    def __init__(self, dim: int, depth: int, channels: int,
                 patch_size=(2, 2), num_heads: int = 4, dim_head: int = 8):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(channels, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)
        self.act2 = nn.ReLU6(inplace=True)

        self.transformer = nn.Sequential(*[
            TransformerLayer(dim, num_heads, dim_head, mlp_ratio=2.0)
            for _ in range(depth)
        ])

        self.conv3 = nn.Conv2d(dim, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.act3 = nn.ReLU6(inplace=True)

        self.conv4 = nn.Conv2d(2 * channels, channels, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)
        self.act4 = nn.ReLU6(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.act1(self.bn1(self.conv1(x)))
        y = self.act2(self.bn2(self.conv2(y)))

        B, C, H, W = y.shape
        y = y.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        y = self.transformer(y)
        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        y = self.act3(self.bn3(self.conv3(y)))
        y = torch.cat([x, y], dim=1)
        y = self.act4(self.bn4(self.conv4(y)))
        return y


class MobileViT(nn.Module):
    """MobileViT XXS built from raw PyTorch layers for SimPhony mapping.

    Config matches the HPAT paper's MobileViT-XXS variant:
      dim=[144, 192, 240], depth=[2, 4, 3], expansion=4
    """

    def __init__(
        self,
        dim=(144, 192, 240),
        depth=(2, 4, 3),
        channels=(16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640),
        image_size=64,
        num_classes=1000,
        expansion=4,
    ):
        super().__init__()
        self.image_size = image_size

        # Stem: Stage 1
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU6(inplace=True),
        )

        # Stage 2
        self.stage2 = InvertedResidual(channels[0], channels[1], stride=1,
                                       expand_ratio=expansion)

        # Stage 3
        self.stage3 = nn.Sequential(
            InvertedResidual(channels[1], channels[2], stride=2, expand_ratio=expansion),
            InvertedResidual(channels[2], channels[3], stride=1, expand_ratio=expansion),
        )

        # Stage 4
        self.stage4 = InvertedResidual(channels[3], channels[4], stride=2,
                                       expand_ratio=expansion)

        # Stage 5: first MobileViT block
        self.stage5 = MobileViTBlock(dim[0], depth[0], channels[5])

        # Stage 6
        self.stage6 = InvertedResidual(channels[5], channels[6], stride=2,
                                       expand_ratio=expansion)

        # Stage 7: second MobileViT block
        self.stage7 = MobileViTBlock(dim[1], depth[1], channels[7])

        # Stage 8
        self.stage8 = InvertedResidual(channels[7], channels[8], stride=2,
                                       expand_ratio=expansion)

        # Stage 9: third MobileViT block
        self.stage9_block = MobileViTBlock(dim[2], depth[2], channels[9])

        # Head
        self.conv_head = nn.Sequential(
            nn.Conv2d(channels[9], channels[10], 1, bias=False),
            nn.BatchNorm2d(channels[10]),
            nn.ReLU6(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[10], num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_stem(x)       # Stage 1
        x = self.stage2(x)          # Stage 2
        x = self.stage3(x)          # Stage 3
        x = self.stage4(x)          # Stage 4
        x = self.stage5(x)          # Stage 5 (MobileViT block)
        x = self.stage6(x)          # Stage 6
        x = self.stage7(x)          # Stage 7 (MobileViT block)
        x = self.stage8(x)          # Stage 8
        x = self.stage9_block(x)    # Stage 9 (MobileViT block)
        x = self.conv_head(x)       # 1x1 -> 640
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MobileViT XXS at full 256x256 resolution (matches HPAT paper evaluation)
    image_size = 256
    model = MobileViT(image_size=image_size, num_classes=1000).to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MobileViT XXS parameters: {total_params:,} (image_size={image_size})")

    # List all layers that SimPhony will detect
    print("\n--- Layers detected by SimPhony ---")
    for name, layer in model.named_modules():
        layer_type = layer.__class__.__name__
        if layer_type in ("Conv2d", "Linear", "MatMul"):
            if isinstance(layer, nn.Conv2d):
                shape = (layer.out_channels, layer.in_channels, *layer.kernel_size)
            elif isinstance(layer, nn.Linear):
                shape = (layer.out_features, layer.in_features)
            else:
                shape = "(matmul)"
            print(f"  {name:50s} {layer_type:15s} {shape}")

    # Config paths
    onn_conversion_cfg = None  # no ONN conversion (raw PyTorch)
    nn_conversion_cfg = "configs/nn_mapping/hpat_cnn.yml"
    model2arch_map_cfg = "configs/architecture_mapping/hpat_cnn.yml"
    arch_cfg_file = "configs/design/architectures/HPAT_hetero.yml"

    print(f"\n--- Simulator configuration ---")
    print(f"  Architecture: {arch_cfg_file}")
    print(f"  Model->Arch map: {model2arch_map_cfg}")
    print(f"  NN precision: {nn_conversion_cfg}")

    sim = ONNArchSimulator(
        nn_model=model,
        onn_conversion_cfg=onn_conversion_cfg,
        nn_conversion_cfg=nn_conversion_cfg,
        onn_model=None,
        model2arch_map_cfg=model2arch_map_cfg,
        devicelib_root="configs/devices",
        device_cfg_files=["*/*.yml"],
        arch_cfg_file=arch_cfg_file,
        arch_version="v1",
        input_shape=(1, 3, image_size, image_size),
        log_path="log/hpat_mobilevit_256.txt",
    )

    print("\n--- Running simulation pipeline ---")

    # 1. Partition cycles
    partition_cycles = sim.simu_partition_cycles(sim.layer_workloads, sim.layer_sizes)
    sim.log_report(partition_cycles, header="Partition Cycles (iter_N, iter_D, iter_M, N, D, M, fw_W, fw_X)")

    # 2. Insertion loss
    insertion_loss = sim.simu_insertion_loss()
    sim.log_report(insertion_loss, header="Insertion Loss (Breakdown)")

    # 3. Energy (computation only, no memory)
    energy_breakdown, total_energy_dict, computation_latency_dict = sim.simu_energy(
        partition_cycles, insertion_loss
    )
    chip_energy = sim.simu_chip_energy(energy_breakdown)

    # 4. Memory cost (HBM/SRAM latency and energy)
    sub_arch_memory_latency, sub_arch_memory_energy, memory_sim_results = sim.simu_memory_cost(
        partition_cycles
    )
    sim.log_report(sub_arch_memory_latency, header="Memory Latency (s)")
    sim.log_report(sub_arch_memory_energy, header="Memory Energy (pJ)")

    # 5. End-to-end latency (computation + memory)
    end_to_end_latency = sim.simu_latency(
        sub_arch_memory_latency, computation_latency_dict
    )
    sim.log_report(end_to_end_latency, header="End-to-End Latency (s)")

    # Extract HPAT sub-arch values
    hpat_key = "HPAT"

    def _sum_dict_values(d, key=None):
        if key is not None:
            v = d.get(key, 0)
        else:
            v = d
        if isinstance(v, dict):
            return sum(vv for vv in v.values() if isinstance(vv, (int, float)))
        return v if isinstance(v, (int, float)) else 0

    total_energy = _sum_dict_values(total_energy_dict, hpat_key)
    if total_energy == 0:
        total_energy = _sum_dict_values(total_energy_dict)

    comp_latency = _sum_dict_values(computation_latency_dict, hpat_key)
    if comp_latency == 0:
        comp_latency = _sum_dict_values(computation_latency_dict)

    e2e_latency = _sum_dict_values(end_to_end_latency, hpat_key)
    if e2e_latency == 0:
        e2e_latency = _sum_dict_values(end_to_end_latency)

    print(f"\n{'='*60}")
    print(f"  Total Energy:         {total_energy:,.2f} pJ ({total_energy/1e6:.4f} uJ)")
    print(f"  Computation Latency:  {comp_latency:.6e} s")
    print(f"  End-to-End Latency:   {e2e_latency:.6e} s")
    print(f"{'='*60}")

    sim.log_report(energy_breakdown, header="Energy Cost Breakdown (pJ)")
    sim.log_report(total_energy_dict, header="Total Energy (pJ)")
    sim.log_report(computation_latency_dict, header="Computation Latency (s)")
    sim.log_report(chip_energy, header="Chip Energy Breakdown (pJ)")

    # 6. Area
    area_breakdown, total_area_raw = sim.simu_area()
    chip_area = sim.simu_chip_area(area_breakdown)

    total_area = _sum_dict_values(total_area_raw, hpat_key)
    if total_area == 0:
        total_area = _sum_dict_values(total_area_raw)

    print(f"\n  Total Area: {total_area:,.2f} um^2 ({total_area/1e6:.4f} mm^2)")

    sim.log_report(area_breakdown, header="Area Cost Breakdown (um^2)")
    sim.log_report(total_area_raw, header="Total Area (um^2)")
    sim.log_report(chip_area, header="Chip Area Breakdown (um^2)")

    print(f"\nLog written to: log/hpat_mobilevit_256.txt")
    print("Done.")


if __name__ == "__main__":
    main()
