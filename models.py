import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    """
    CIFAR-10 (32x32) Generator using Upsample + Conv.
    Typically reduces checkerboard artifacts vs ConvTranspose-only G.
    """
    def __init__(self, z_dim: int = 128, base_ch: int = 256):
        super().__init__()

        # 1x1 -> 4x4
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_ch * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),
        )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        # 4 -> 8 -> 16 -> 32
        self.up1 = up_block(base_ch * 2, base_ch)          # 4 -> 8
        self.up2 = up_block(base_ch, base_ch // 2)         # 8 -> 16
        self.up3 = up_block(base_ch // 2, base_ch // 4)    # 16 -> 32

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base_ch // 4, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.to_rgb(x)

class AutoEncoderDiscriminator(nn.Module):
    """
    BEGAN discriminator: autoencoder with reconstruction loss.
    """
    def __init__(self, base_ch: int = 64, bottleneck: int = 128):
        super().__init__()

        self.enc = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1)),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1)),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.to_bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear((base_ch * 4) * 4 * 4, bottleneck),
            nn.LeakyReLU(0.2, True),
        )

        self.from_bottleneck = nn.Sequential(
            nn.Linear(bottleneck, (base_ch * 4) * 4 * 4),
            nn.LeakyReLU(0.2, True),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        z = self.to_bottleneck(h)
        h2 = self.from_bottleneck(z).view(x.size(0), -1, 4, 4)
        return self.dec(h2)
