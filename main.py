from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import nest_asyncio
import uvicorn
import os
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gdown

# === Init ===
load_dotenv()
nest_asyncio.apply()
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Health check ===
@app.get("/")
def root():
    return {"status": "🟢 API is live and healthy"}

# === Download encoder ONNX if missing ===
encoder_path = "encoder_quantized.onnx"
if not os.path.exists(encoder_path):
    print("📥 Downloading encoder_quantized.onnx from Google Drive...")
    gdown.download(id="1TomEm8_Nf2jicPSt0dqEWA-X0KvIzmA3", output=encoder_path, quiet=False)

# === Load ONNX ===
tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
encoder_session = ort.InferenceSession(encoder_path)
classifier_session = ort.InferenceSession("classifier.onnx")

# === Load scaler and label encoder ===
scaler, label_encoder, _ = joblib.load("classifier_pipeline_light.pkl")

# === FastAPI Schema ===
class QueryInput(BaseModel):
    text: str

# === Encode text using ONNX encoder ===
def encode_text(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }
    outputs = encoder_session.run(None, ort_inputs)
    token_embeddings = outputs[0]
    return scaler.transform(token_embeddings.astype(np.float32))

# === Local classifier ===
@app.post("/classify")
def classify_query(query: QueryInput):
    try:
        emb = encode_text(query.text)
        pred = classifier_session.run(None, {"input": emb})[0]
        label_idx = int(pred[0])
        label = label_encoder.inverse_transform([label_idx])[0]
        print(f"🔍 Final ONNX Output: {label}")
        return {"label": label}
    except Exception as e:
        return {"error": str(e)}

# === OpenAI GPT ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/ask")
async def ask_gpt(query: QueryInput):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful healthcare assistant of asha or anm that replies in Hindi."},
                {"role": "user", "content": query.text},
            ],
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"⚠️ GPT error: {str(e)}"}

# === Local Run ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))