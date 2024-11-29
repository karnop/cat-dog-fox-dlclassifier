import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from components.model_architecture import CNNModel  # Your CNN model code
from io import BytesIO
from PIL import Image
from artifacts.artifacts import saved_model_path

# Initialize the FastAPI app
app = FastAPI()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple transform for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the expected input size of the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Load the pre-trained model
model_path = saved_model_path  # Update this path to your saved model
model = CNNModel(num_classes=3)  # Update this if your model class differs
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # Set the model to evaluation mode

# Map the class indices to labels
class_labels = {0: 'cat', 1: 'dog', 2: 'fox'}

# Define a helper function to load and preprocess an image
def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Define the route for the prediction
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        # Run the model prediction
        with torch.no_grad():  # Don't calculate gradients for inference
            outputs = model(image)
            _, predicted_class = torch.max(outputs, 1)

        # Get the predicted label
        predicted_label = class_labels[predicted_class.item()]

        # Return the prediction
        return {"prediction": predicted_label}
    except Exception as e:
        return {"error": str(e)}

