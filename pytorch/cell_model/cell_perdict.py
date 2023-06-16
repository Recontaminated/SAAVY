import torch
from torchvision import transforms
from PIL import Image
from cell_train import CellViabilityNet




# Constants
IMG_SIZE = 128


# Create data transforms for the cell image dataset (same as in your training script)
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the trained model
model = CellViabilityNet()
model.load_state_dict(torch.load('cell_viability_model.pth'))
model.eval()

# Load the test image
test_image_path = r"C:\Users\minec\OneDrive\Documents\GitHub\kylieDataAnylasis\pytorch\cell_model\dataset\166.png"
image = Image.open(test_image_path)



def get_prediction(image):
# Apply data transformations
    image = data_transforms(image)

    # Add the batch dimension
    image = image.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(image)
        cell_viability_score = output.item()

    return cell_viability_score