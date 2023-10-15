import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data preprocessing
transform = transforms.Compose([transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    './data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True, num_workers=16)

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

# Define the loss function (e.g., negative log-likelihood)
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)  # Move data to GPU and flatten
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'glow_mnist_model.pth')
