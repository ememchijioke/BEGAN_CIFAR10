import os
import argparse
import torch

from models import Generator
from utils import get_device, ensure_dir, save_sample_grid, EMA


def parse_args():
    p = argparse.ArgumentParser("Sample from trained BEGAN Generator")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="./outputs/sample_grid.png")
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--g-ch", type=int, default=128)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--ema", action="store_true", help="use EMA weights if checkpoint contains them")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location=device)

    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # optionally apply EMA if present
    if args.ema and "ema" in ckpt:
        ema_info = ckpt["ema"]
        ema = EMA(G, decay=ema_info.get("decay", 0.999))
        # overwrite shadow with saved values
        ema.shadow = ema_info["shadow"]
        ema.apply_to(G)

    z = torch.randn(args.n, args.z_dim, 1, 1, device=device)
    with torch.no_grad():
        imgs = G(z)

    ensure_dir(os.path.dirname(args.out) or ".")
    save_sample_grid(imgs, args.out, nrow=8)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
