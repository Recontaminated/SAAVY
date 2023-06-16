import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from PIL import Image
# Constants
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4

# Define the CNN architecture
class CellViabilityNet(nn.Module):

    def __init__(self):
        super(CellViabilityNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)# Takes input image with 3 channels RGB and outouts 32 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * IMG_SIZE // 4 * IMG_SIZE // 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * IMG_SIZE // 4 * IMG_SIZE // 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create data transforms for the cell image dataset
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,dataset_folder):
        files = os.listdir(dataset_folder)
        files = [file[:-4] for file in files]
        files = [int(file) for file in files if file != 'data']
        files.sort()
        files = [dataset_folder+"/"+str(file)+".png" for file in files]
        self.files = files
        with open(dataset_folder+"/data.txt") as f:
            self.labels = f.readlines()
        #normalize labels
        self.labels = [float(label)/100 for label in self.labels]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx])
        img = data_transforms(img)
        label = float(self.labels[idx])
        return img, torch.tensor(label, dtype=torch.float32)
    



if __name__ == "__main__":
    train_dataset = CustomDataset("./dataset")
    # Load the dataset
    val_dataset = CustomDataset("./dataset")



    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instantiate the cell viability model, loss function, and optimizer
    model = CellViabilityNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    def train(model, loader, criterion, optimizer, device):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation loop
    def validate(model, loader, criterion, device):
        model.eval()
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)

            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item()
        return total_loss / len(loader)

    # Main training and validation loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        
        train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"Validation Loss: {val_loss:.4f}\n")

    # Save the trained model
    torch.save(model.state_dict(), 'cell_viability_model.pth')
