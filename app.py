from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
import numpy as np
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

app = FastAPI()

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for frontend HTML/JS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your trained model
model = load_model("model.h5")

# Load class indices mapping
idx_to_class = {
    0: "የበሰለ ፖም",
    1: "የበሰለ ሙዝ",
    2: "የበሰለ ብርቱካን",
    3: "የበሰበሰ ፖም",
    4: "የበሰበሰ ሙዝ",
    5: "የበሰበሰ ብርቱካን",
    6: "ያልበሰለ ፖም",
    7: "ያልበሰለ ሙዝ",
    8: "ያልበሰለ ብርቱካን"
}
# Preprocess image function
def preprocess_image(img: Image.Image, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_array = np.array(img.resize(target_size))
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Endpoint for file upload
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))
    processed_img = preprocess_image(img)
    pred = model.predict(processed_img)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    pred_class_name = idx_to_class[pred_class_idx]
    return JSONResponse({"class": pred_class_name})

# Endpoint for camera capture (send base64 or file)
@app.post("/predict_camera")
async def predict_camera(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(BytesIO(img_bytes))
    processed_img = preprocess_image(img)
    pred = model.predict(processed_img)
    pred_class_idx = np.argmax(pred, axis=1)[0]
    pred_class_name = idx_to_class[pred_class_idx]
    return JSONResponse({"class": pred_class_name})
