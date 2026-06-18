"""Display model architecture, parameters, and FLOPs for MammaSegNet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nets.segmentors.breastseg_multitask import MammaSegNet


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(num: int) -> str:
    """Format large numbers with K/M/G suffixes."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def print_model_info(model: torch.nn.Module, phi: str, input_size: tuple[int, int]) -> None:
    """Print model architecture information."""
    total_params, trainable_params = count_parameters(model)
    
    print("=" * 80)
    print(f"MammaSegNet Model Summary (Backbone: MiT-{phi.upper()})")
    print("=" * 80)
    print(f"\n📊 Model Statistics:")
    print(f"  • Total Parameters:     {format_number(total_params):>10} ({total_params:,})")
    print(f"  • Trainable Parameters: {format_number(trainable_params):>10} ({trainable_params:,})")
    print(f"  • Input Size:           {input_size[0]} × {input_size[1]}")
    print(f"  • Number of Classes:    2 (background, breast)")
    
    print(f"\n🏗️ Model Architecture:")
    print(f"  • Backbone:             MammaSeg MiT-{phi.upper()}")
    print(f"  • Segmentation Head:    MammaSeg Decode Head")
    print(f"  • Detection Head:       FCOS (Anchor-free)")
    print(f"  • Feature Pyramid:      FPN")
    
    # Print layer-by-layer breakdown
    print(f"\n📋 Layer Breakdown:")
    print(f"  {'Component':<30} {'Parameters':>15} {'Trainable':>15}")
    print(f"  {'-' * 30} {'-' * 15} {'-' * 15}")
    
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if module_params > 0:
            print(f"  {name:<30} {format_number(module_params):>15} {format_number(module_trainable):>15}")
    
    print("=" * 80)


def estimate_flops(model: torch.nn.Module, input_size: tuple[int, int], device: torch.device) -> None:
    """Estimate FLOPs using thop library if available."""
    try:
        from thop import clever_format, profile
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # Temporarily disable detection head for FLOPs calculation
        # (since it requires additional inputs)
        model.eval()
        with torch.no_grad():
            flops, params = profile(model.backbone, (dummy_input,), verbose=False)
        
        # Multiply by 2 following YOLOX convention
        # (counting both multiply and add as separate operations)
        flops = flops * 2
        flops_str, params_str = clever_format([flops, params], "%.3f")
        
        print(f"\n⚡ Computational Complexity (Backbone only):")
        print(f"  • FLOPs:  {flops_str}")
        print(f"  • Params: {params_str}")
        print(f"\n  Note: Full model FLOPs include segmentation head, detection head, and FPN.")
        
    except ImportError:
        print(f"\n⚠️  Install 'thop' to calculate FLOPs: pip install thop")


def main() -> None:
    parser = argparse.ArgumentParser(description="Display MammaSegNet model summary.")
    parser.add_argument(
        "--phi",
        type=str,
        default="b2",
        choices=["b0", "b1", "b2", "b3", "b4", "b5"],
        help="MammaSeg backbone variant (default: b2)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Input image size (default: 512 512)",
    )
    parser.add_argument(
        "--flops",
        action="store_true",
        help="Calculate FLOPs (requires thop library)",
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build model
    model = MammaSegNet(
        num_classes=2,
        phi=args.phi,
        init_from=None,
        use_prior=True,
    ).to(device)
    
    # Print model information
    print_model_info(model, args.phi, tuple(args.input_size))
    
    # Optionally calculate FLOPs
    if args.flops:
        estimate_flops(model, tuple(args.input_size), device)
    
    print()


if __name__ == "__main__":
    main()

