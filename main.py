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
from huggingface_hub import hf_hub_download
# import psutil
import threading
import time

# === Init ===
load_dotenv()
nest_asyncio.apply()
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Globals (for lazy loading) ===
encoder_session = None
tokenizer = None

# === Constants ===
CLASSIFIER_PATH = "classifier.onnx"
TOKENIZER_PATH = "tokenizer/"
HUB_REPO_ID = "panghal/swasthya-encoder"
HUB_FILENAME = "encoder_quantized.onnx"

# === Load classifier pipeline ===
scaler, label_encoder, _ = joblib.load("classifier_pipeline_light.pkl")
classifier_session = ort.InferenceSession(CLASSIFIER_PATH)

# === Schema ===
class QueryInput(BaseModel):
    text: str

# === Lazy ONNX encoder loader ===
def get_encoder():
    global encoder_session, tokenizer

    if encoder_session is None:
        print("📥 Downloading encoder_quantized.onnx from Hugging Face...")
        encoder_path = hf_hub_download(repo_id=HUB_REPO_ID, filename=HUB_FILENAME)
        print("📦 Loading encoder ONNX model...")
        encoder_session = ort.InferenceSession(encoder_path)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    return encoder_session, tokenizer

# === Encode text ===
def encode_text(text):
    encoder, tokenizer = get_encoder()

    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    outputs = encoder.run(None, ort_inputs)
    token_embeddings = outputs[0]
    return scaler.transform(token_embeddings.astype(np.float32))

# === Classifier endpoint ===
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

# === OpenAI GPT endpoint ===
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

# === Health check ===
@app.get("/")
def root():
    return {"status": "🟢 API is live and healthy"}

# === Local run ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))