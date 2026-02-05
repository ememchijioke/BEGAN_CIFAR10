import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_args():
    p = argparse.ArgumentParser("Plot BEGAN training curves")
    p.add_argument("--log", type=str, required=True, help="path to train_log.csv")
    p.add_argument("--out", type=str, default="training_curves.png")
    return p.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.log)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("BEGAN Training Dynamics (CIFAR-10)", fontsize=14)

    axs[0, 0].plot(df["L_real"], label="L_real")
    axs[0, 0].set_title("Reconstruction Loss (Real)")
    axs[0, 0].grid(True)

    axs[0, 1].plot(df["L_fake"], label="L_fake", color="orange")
    axs[0, 1].set_title("Reconstruction Loss (Fake)")
    axs[0, 1].grid(True)

    axs[1, 0].plot(df["k_t"], label="k_t", color="green")
    axs[1, 0].set_title("Equilibrium Term k_t")
    axs[1, 0].grid(True)

    axs[1, 1].plot(df["M_convergence"], label="M", color="red")
    axs[1, 1].set_title("Convergence Measure M")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(args.out)
    plt.close()

    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
