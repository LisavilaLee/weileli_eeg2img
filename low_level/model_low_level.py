import torch.nn as nn


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=1024, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        return x


class encoder_low_level(nn.Module):
    def __init__(self, num_channels=63, sequence_length=250, num_subjects=1, num_features=64, num_latents=1024,
                 num_blocks=1):
        super(encoder_low_level, self).__init__()
        self.subject_wise_linear = nn.ModuleList([nn.Linear(sequence_length, 128) for _ in range(num_subjects)])

        # CNN upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(8064, 1024, kernel_size=4, stride=2, padding=1),  # (1, 1) -> (2, 2)
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # (2, 2) -> (4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (4, 4) -> (8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (8, 8) -> (16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (16, 16) -> (32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 32) -> (64, 64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),  # Keep size (64, 64)
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, kernel_size=1, stride=1, padding=0),  # Output shape (4, 64, 64)
        )

    def forward(self, x):
        # Apply subject-wise linear layer
        x = self.subject_wise_linear[0](x)  # Output shape: (batchsize, 63, 128)
        # Reshape to match the input size for the upsampler
        x = x.view(x.size(0), 8064, 1, 1)  # Reshape to (batch_size, 8064, 1, 1)
        out = self.upsampler(x)  # Pass through the upsampler
        return out
