from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
import io
import random
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="ORIONIS API", description="Includes Quiz + Constellations Engine")

# Allow requests from your frontend origin
origins = [
    "https://orionis-backend.onrender.com/",  # React Native Web
    "orionis-backend:10000",
    "http://44.229.227.142:10000" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (not recommended for prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------
# YOLO Detection Model
# --------------------------
model = YOLO("detectionEngine.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference
        results = model.predict(img)

        # Extract results
        boxes = results[0].boxes
        predictions = []
        for box in boxes:
            prediction = {
                "xmin": float(box.xyxy[0][0]),
                "ymin": float(box.xyxy[0][1]),
                "xmax": float(box.xyxy[0][2]),
                "ymax": float(box.xyxy[0][3]),
                "confidence": float(box.conf[0]),
                "class": str(box.cls[0]),
                "name": model.names[int(box.cls[0])]
            }
            predictions.append(prediction)

        return JSONResponse(content=prediction, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --------------------------
# Astronomy Quiz API
# --------------------------

with open("question_bank.json", "r") as f:
    questions = json.load(f)

@app.get("/")
def home():
    return {
        "message": "Welcome to the Astronomy API 🚀",
        "endpoints": ["/predict/", "/questions", "/questions/random", "/questions/{id}"]
    }

@app.get("/questions")
def get_questions():
    return questions

@app.get("/questions/random")
def get_random_question():
    return random.choice(questions)

@app.get("/questions/{question_id}")
def get_question(question_id: int):
    for q in questions:
        if q["id"] == question_id:
            return q
    return {"error": "Question not found"}
