from fastapi import FastAPI, UploadFile
import time
import logging
from src.inference import load_model, predict

app = FastAPI()
model = load_model()

logging.basicConfig(level=logging.INFO)
request_count = 0

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_image(file: UploadFile):

    global request_count
    request_count += 1

    start = time.time()
    contents = await file.read()
    result = predict(contents, model)
    duration = time.time() - start

    logging.info(f"Prediction took {duration:.4f}s | Total requests: {request_count}")

    return result
