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

# TODO: replace this with loading server model, images, embeddings, etc.
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
    
    # TODO: outfit recommendation, return descriptions
    # feel free to ignore this part for now. we are probably more interested in end-to-end
    print("image loading finish", time.time())

    image = image.resize((224, 224))
    result = resnet_and_knn(image)

    return JSONResponse(content=result)

# endpoint for our system
@app.post("/recommendation_only/")
async def recommendation_only(embedding: List[float] = Body(...), content_length: int = Header(...)):  
    try:
        # TODO: outfit recommendation, return descriptions
        process_start = time.time()
        result = knn(embedding)
        
        process_end = time.time()
        # print(process_start, process_end)

        return JSONResponse(content={"result": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# endpoint for testing the server is up
@app.get("/alive/")
async def liveness():
    return True
