import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import msvcrt

class PairedGenerator(nn.Module):
    def __init__(self):
        super(PairedGenerator, self).__init__()
        # First convolution now accepts 6 channels
        self.initial = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling blocks
        self.down_blocks = nn.Sequential(
            self._down_block(64, 128),
            self._down_block(128, 256)
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)]
        )

        # Upsampling blocks
        self.up_blocks = nn.Sequential(
            self._up_block(256, 128),
            self._up_block(128, 64)
        )

        # Output convolution
        self.output = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )

    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_blocks(x)
        return self.output(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # First layer (no normalization)
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Subsequent layers with normalization
            self._discriminator_block(64, 128),
            self._discriminator_block(128, 256),
            self._discriminator_block(256, 512),

            # Final layer to output a prediction
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def _discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class PairedStyleDataset(Dataset):
    def __init__(self, shirt_dir, art_dir, transform=None):
        self.shirt_files = sorted(os.listdir(shirt_dir))
        self.art_files = sorted(os.listdir(art_dir))

        # Ensure equal number of images
        min_len = min(len(self.shirt_files), len(self.art_files))
        self.shirt_files = self.shirt_files[:min_len]
        self.art_files = self.art_files[:min_len]

        self.shirt_dir = shirt_dir
        self.art_dir = art_dir
        self.transform = transform

    def __len__(self):
        return len(self.shirt_files)

    def __getitem__(self, idx):
        # Load shirt and art images
        shirt_path = os.path.join(self.shirt_dir, self.shirt_files[idx])
        art_path = os.path.join(self.art_dir, self.art_files[idx])

        shirt_img = Image.open(shirt_path).convert('RGB')
        art_img = Image.open(art_path).convert('RGB')

        # Apply transforms if specified
        if self.transform:
            shirt_img = self.transform(shirt_img)
            art_img = self.transform(art_img)

        # Concatenate images along channel dimension
        combined_input = torch.cat([shirt_img, art_img], dim=0)

        return {
            'input': combined_input,  # 6-channel tensor
            'target': shirt_img  # Original shirt image as target
        }
    


# Inference function
def apply_style_transfer(shirt_img, art_img, generator, transform):
    # Load and preprocess images
    
    os.write(1,f"=================================Entered in funcion===============================================".encode())
    # shirt_img = Image.open(shirt_path).convert('RGB')
    # art_img = Image.open(art_path).convert('RGB')

    shirt_img = Image.fromarray(shirt_img)
    art_img = Image.fromarray(art_img)
    os.write(1,f"=================================OPening===============================================".encode())
    # Apply transforms
    shirt_tensor = transform(shirt_img).unsqueeze(0)
    art_tensor = transform(art_img).unsqueeze(0)

    # Concatenate images
    combined_input = torch.cat([shirt_tensor, art_tensor], dim=1)

    # Generate styled shirt
    with torch.no_grad():
        styled_shirt = generator(combined_input)

    # Convert to image
    to_pil = transforms.ToPILImage()
    styled_shirt_img = to_pil(styled_shirt.squeeze(0))
    os.write(1,f"=================================returned shirt===============================================".encode())
    
    return styled_shirt_img


def merger(shirt_img, art_img):

    #Inference
    generator = PairedGenerator()
    #generator.load_state_dict(torch.load('checkpoint_epoch_200.pth')['generator_state_dict'])
    generator.load_state_dict(torch.load('checkpoint_epoch_200.pth', map_location=torch.device('cpu'))['generator_state_dict'])
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    os.write(1,f"================================={len(shirt_img), shirt_img.ndim}===============================================".encode())
    styled_shirt = apply_style_transfer(shirt_img, art_img, generator, transform)

    return styled_shirt
