from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

# Load model
model = YOLO("detectionEngine.pt")

@app.post("/predict/")
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
                "class": int(box.cls[0])
            }
            predictions.append(prediction)

        return JSONResponse(content=predictions)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

