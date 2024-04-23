import time
server_start_time = time.time()

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Header
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware  
from PIL import Image

# sameple_run is like a utility file. feel free to ignore it
from app.sample_run import load_model, resnet_and_knn, knn

from typing import List  
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

server_loading_done = time.time()
print("Server start up time :", server_loading_done - server_start_time, "seconds")

# endpoint for running everything on server
@app.post("/full_prediction/")
async def predict(file: UploadFile = File(...), content_length: int = Header(...)):

    print(time.time())
    process_start = time.time()
    
    image_data = await file.read()
    # clothes image uploaded by the user
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    print("image loading finish", time.time())
    
    result = resnet_and_knn(image)

    return JSONResponse(content=result)

# endpoint for our system
@app.post("/recommendation_only/")
async def recommendation_only(embedding: List[List[List[List[float]]]] = Body(...), content_length: int = Header(...)):  
    try:
        # TODO: outfit recommendation, return descriptions
        print("Received embeddings: ", time.time())
        
        process_start = time.time()
        images, descriptions = knn(embedding)
        process_end = time.time()

        # print(process_start, process_end)
        return JSONResponse(content={"images": images, "descriptions": descriptions})

    except Exception as e:
        error_words = str(e).split()
        # Get the first 100 words
        short_error_message = ' '.join(error_words[:100])
        raise HTTPException(status_code=500, detail=short_error_message)

# endpoint for testing the server is up
@app.get("/alive/")
async def liveness():
    return True
