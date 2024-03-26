from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mock function to load a pre-trained LeNet model
def load_model():
    model = "some pretrained model"  # Placeholder for the actual model loading logic
    return model

model = load_model()

# Define the image transform
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    
    # Mock inference
    path_to_dummy_image = './images/dummy_outfit.jpg'
    # Return the dummy image as a response
    return FileResponse(path=path_to_dummy_image, media_type='image/png')
