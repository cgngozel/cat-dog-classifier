from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()


model = tf.keras.models.load_model('cat_dog_model.keras')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content)).convert("RGB")
    
    # görseli modelin boyutu olan 128x128 yap ve arraye çevir
    image = image.resize((128, 128))
    img_array = np.array(image)
    

    img_array = img_array / 255.0
    
    img_array = np.expand_dims(img_array, axis=0)
    
    tahmin = model.predict(img_array)
    skor = float(tahmin[0][0])
    
    if skor < 0.5:
        etiket = "CAT 🐱"
        olasilik = (1 - skor) * 100
    else:
        etiket = "DOG 🐶"
        olasilik = skor * 100
        
    return {
        "label": etiket,
        "confidence": round(olasilik, 2)
    }

# Çalıştırmak için terminale: uvicorn main:app --reload