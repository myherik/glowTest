import torch
import torch.nn as nn
from torchvision import datasets, transforms
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Load the trained model
# Define a Glow-like architecture

class GlowBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(GlowBlock, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels)
        )

    def forward(self, x):
        return self.net(x)

class Glow(nn.Module):
    def __init__(self, num_blocks, in_channels, mid_channels):
        super(Glow, self).__init__()
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.blocks = nn.ModuleList([GlowBlock(in_channels, mid_channels) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Instantiate the model and move it to the GPU
num_blocks = 3
in_channels = 28 * 28  # MNIST images are 28x28
mid_channels = 64
model = Glow(num_blocks, in_channels, mid_channels).to(device)
model.load_state_dict(torch.load('glow_mnist_model.pth'))
model.eval()  # Set the model to evaluation mode

transform = transforms.Compose([transforms.ToTensor()])
test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    './data', train=False, transform=transform),
    batch_size=64, shuffle=True, num_workers=16)

# Choose 10 random test samples
random.seed(42)  # Set a seed for reproducibility
sample_indices = random.sample(range(len(test_loader.dataset)), 10)

# Initialize lists to store input and output images
input_images = []
output_images = []


import matplotlib.pyplot as plt

for idx in sample_indices:
    data = test_loader.dataset[idx][0].view(1, -1).to(device)
    output = model(data)

    input_image = data.view(28, 28).cpu().detach().numpy()
    output_image = output.view(28, 28).cpu().detach().numpy()

    input_images.append(input_image)
    output_images.append(output_image)

# Display input and output images
plt.figure(figsize=(14, 6))

for i in range(10):
    plt.subplot(2, 10, i + 1)
    plt.imshow(input_images[i], cmap='gray')
    plt.axis('off')
    plt.title("Input")

    plt.subplot(2, 10, i + 11)
    plt.imshow(output_images[i], cmap='gray')
    plt.axis('off')
    plt.title("Output")

plt.show()