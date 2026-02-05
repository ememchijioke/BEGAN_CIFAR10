import os
import csv
import argparse
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from tqdm import tqdm

from models import Generator, AutoEncoderDiscriminator, weights_init
from utils import (
    set_seed, get_device, ensure_dir, timestamp, save_config,
    save_sample_grid, EMA
)


def parse_args():
    p = argparse.ArgumentParser("BEGAN CIFAR-10 Trainer (PyTorch)")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./outputs/runs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-decay-epoch", type=int, default=80, help="start decaying LR after this epoch")
    p.add_argument("--lr-decay-factor", type=float, default=0.5, help="multiply LR by this factor at each decay step")
    p.add_argument("--lr-decay-every", type=int, default=40, help="decay interval in epochs")


    # TTUR
    p.add_argument("--lr-g", type=float, default=2e-4, help="generator lr")
    p.add_argument("--lr-d", type=float, default=1e-4, help="discriminator lr")

    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--g-ch", type=int, default=256)      # strong G capacity (works well with Upsample+Conv G)
    p.add_argument("--d-ch", type=int, default=64)
    p.add_argument("--d-bottleneck", type=int, default=256)  # bigger AE bottleneck for better manifold modeling

    # BEGAN-specific
    p.add_argument("--gamma", type=float, default=0.7, help="equilibrium hyperparam gamma")
    p.add_argument("--lambda-k", type=float, default=1e-3, help="k_t update rate")
    p.add_argument("--k-init", type=float, default=0.0)

    # update ratio (new)
    p.add_argument("--d-steps", type=int, default=2, help="discriminator updates per batch")
    p.add_argument("--g-steps", type=int, default=1, help="generator updates per batch")

    # optional EMA enhancement
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.999)

    # logging/saving
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1, help="save samples/checkpoints every N epochs")
    p.add_argument("--sample-n", type=int, default=64, help="number of samples for grid")
    p.add_argument("--tag", type=str, default="", help="short label for the run")
    p.add_argument("--author", type=str, default="Emem Chijioke")
    p.add_argument("--course", type=str, default="Lab: Neural Networks and Deep Learning")
    return p.parse_args()


def make_dataloaders(data_dir: str, batch_size: int, num_workers: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
    ])

    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)

    val_size = 5000
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader


def save_checkpoint(path: str, G, D, optG, optD, epoch: int, k_t: float, ema_state: Dict[str, Any] | None):
    ckpt = {
        "epoch": epoch,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "k_t": k_t,
    }
    if ema_state is not None:
        ckpt["ema"] = ema_state
    torch.save(ckpt, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    tag_part = f"_{args.tag}" if args.tag else ""
    run_name = args.run_name or f"began_emem_{timestamp()}{tag_part}"
    run_dir = os.path.join(args.out_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")
    log_dir = os.path.join(run_dir, "logs")
    ensure_dir(ckpt_dir)
    ensure_dir(sample_dir)
    ensure_dir(log_dir)

    cfg = vars(args)
    cfg["device"] = str(device)
    cfg["project_info"] = {
        "author": args.author,
        "course": args.course,
        "model": "BEGAN (Boundary Equilibrium GAN)",
        "dataset": "CIFAR-10 (32x32)",
        "framework": "PyTorch",
        "notes": "Upsample+Conv G, TTUR, EMA, D:G update ratio"
    }
    save_config(os.path.join(run_dir, "config.json"), cfg)

    train_loader, _ = make_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    # Models
    G = Generator(z_dim=args.z_dim, base_ch=args.g_ch).to(device)
    D = AutoEncoderDiscriminator(base_ch=args.d_ch, bottleneck=args.d_bottleneck).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    recon_loss = nn.L1Loss(reduction="mean")

    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    ema = None
    if args.ema:
        ema = EMA(G, decay=args.ema_decay)

    fixed_z = torch.randn(args.sample_n, args.z_dim, 1, 1, device=device)

    k_t = float(args.k_init)

    # CSV log
    csv_path = os.path.join(log_dir, "train_log.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "iter", "L_real", "L_fake", "L_D", "L_G", "k_t", "M_convergence"])

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        
        # LR decay schedule (piecewise)
        if epoch >= args.lr_decay_epoch and ((epoch - args.lr_decay_epoch) % args.lr_decay_every == 0):
            for pg in optG.param_groups:
                pg["lr"] *= args.lr_decay_factor
            for pg in optD.param_groups:
                pg["lr"] *= args.lr_decay_factor
            print(f"[lr_decay] epoch={epoch} lr_g={optG.param_groups[0]['lr']:.2e} lr_d={optD.param_groups[0]['lr']:.2e}")
        G.train()
        D.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            bs = x.size(0)

            # -------------------------
            # Train D multiple steps
            # -------------------------
            for _ in range(args.d_steps):
                z = torch.randn(bs, args.z_dim, 1, 1, device=device)
                x_fake = G(z).detach()

                D_x = D(x)
                D_g = D(x_fake)

                L_real = recon_loss(D_x, x)
                L_fake = recon_loss(D_g, x_fake)

                L_D = L_real - (k_t * L_fake)

                optD.zero_grad(set_to_none=True)
                L_D.backward()
                optD.step()

                # Update k_t using latest losses
                k_t = k_t + args.lambda_k * (args.gamma * L_real.item() - L_fake.item())
                k_t = max(0.0, min(1.0, k_t))

            # -------------------------
            # Train G multiple steps
            # -------------------------
            for _ in range(args.g_steps):
                z2 = torch.randn(bs, args.z_dim, 1, 1, device=device)
                x_gen = G(z2)
                D_gen = D(x_gen)
                L_G = recon_loss(D_gen, x_gen)

                optG.zero_grad(set_to_none=True)
                L_G.backward()
                optG.step()

                if ema is not None:
                    ema.update(G)

            # Convergence measure M
            M = L_real.item() + abs(args.gamma * L_real.item() - L_fake.item())

            global_step += 1
            pbar.set_postfix({"Lr": f"{L_real.item():.3f}", "Lf": f"{L_fake.item():.3f}", "k": f"{k_t:.3f}", "M": f"{M:.3f}"})

            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, global_step, L_real.item(), L_fake.item(), L_D.item(), L_G.item(), k_t, M])

        # Save samples + checkpoint each epoch
        if (epoch % args.save_every) == 0:
            # sample with EMA if enabled (copy weights into temp model)
            if ema is not None:
                G_eval = Generator(z_dim=args.z_dim, base_ch=args.g_ch).to(device)
                G_eval.load_state_dict(G.state_dict())
                ema.apply_to(G_eval)
                G_eval.eval()
            else:
                G_eval = G
                G_eval.eval()

            with torch.no_grad():
                samples = G_eval(fixed_z)

            save_sample_grid(samples, os.path.join(sample_dir, f"samples_epoch_{epoch:03d}.png"), nrow=8)

            ema_state = None
            if ema is not None:
                ema_state = {"decay": args.ema_decay, "shadow": ema.shadow}

            save_checkpoint(
                os.path.join(ckpt_dir, f"ckpt_epoch_{epoch:03d}.pt"),
                G, D, optG, optD, epoch, k_t, ema_state
            )

    print(f"Done. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
