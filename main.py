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
ENCODER_PATH = "encoder_quantized.onnx"
CLASSIFIER_PATH = "classifier.onnx"
TOKENIZER_PATH = "tokenizer/"
GDRIVE_ID = "1TomEm8_Nf2jicPSt0dqEWA-X0KvIzmA3"

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
        if not os.path.exists(ENCODER_PATH):
            print("📥 Downloading encoder_quantized.onnx from Google Drive...")
            gdown.download(id=GDRIVE_ID, output=ENCODER_PATH, quiet=False)
        print("📦 Loading encoder ONNX model...")
        encoder_session = ort.InferenceSession(ENCODER_PATH)

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

# === Memory monitor thread ===
# def log_memory_usage():
#     process = psutil.Process(os.getpid())
#     while True:
#         mem = process.memory_info().rss / (1024 * 1024)
#         print(f"🧠 Current RAM Usage: {mem:.2f} MB")
#         time.sleep(10)

# threading.Thread(target=log_memory_usage, daemon=True).start()

# === Health check ===
@app.get("/")
def root():
    return {"status": "🟢 API is live and healthy"}

# === Local run ===
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))