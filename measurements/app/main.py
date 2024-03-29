from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from PIL import Image
from app.sample_run import load_model, resnet_and_knn, knn
from typing import List  # Import List from typing
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

load_model()

@app.post("/full_prediction/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    result = resnet_and_knn(image)

    return JSONResponse(content=result)

@app.post("/recommendation_only/")
async def recommendation_only(embedding: List[float] = Body(...)):  # Adjust typing as necessary
    try:
        result = knn(embedding)
        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
