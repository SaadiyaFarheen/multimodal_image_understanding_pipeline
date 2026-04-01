from fastapi import FastAPI, UploadFile, File
import shutil

from predict import predict_image
from caption import generate_caption

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    file_path = f"temp_{file.filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    labels = predict_image(file_path)
    caption = generate_caption(file_path)

    return {
        "labels": labels,
        "caption": caption
    }