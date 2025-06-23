from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
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
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse
from transformers import PreTrainedTokenizerFast

# === Init ===
load_dotenv()
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Globals ===
encoder_session = None
tokenizer = None

# === Constants ===
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "classifier.onnx")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "classifier_pipeline_light.pkl")
HUB_REPO_ID = "panghal/swasthya-encoder"
HUB_FILENAME = "encoder_quantized.onnx"

# === Load classifier pipeline ===
print("📦 Loading classifier pipeline...")
scaler, label_encoder, _ = joblib.load(PIPELINE_PATH)
classifier_session = ort.InferenceSession(CLASSIFIER_PATH)

# === Schema ===
class QueryInput(BaseModel):
    text: str

# === Lazy encoder loader ===
def get_encoder():
    global encoder_session, tokenizer

    if encoder_session is None:
        print("📥 Downloading encoder from HuggingFace Hub...")
        encoder_path = hf_hub_download(repo_id=HUB_REPO_ID, filename=HUB_FILENAME)
        print("📦 Loading encoder ONNX model...")
        encoder_session = ort.InferenceSession(encoder_path)

    if tokenizer is None:
        print("🧠 Loading tokenizer...")
        TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(TOKENIZER_PATH, "tokenizer.json"))

    return encoder_session, tokenizer

# === Encode Text ===
def encode_text(text):
    encoder, tokenizer = get_encoder()
    print("🔤 Tokenizing text...")
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    print("🧮 Running encoder ONNX model...")
    outputs = encoder.run(None, ort_inputs)
    token_embeddings = outputs[0]
    print("📐 Embedding shape:", token_embeddings.shape)
    return scaler.transform(token_embeddings.astype(np.float32))

# === Classifier Endpoint ===
@app.post("/classify")
def classify_query(query: QueryInput):
    try:
        print("📝 Received text:", query.text)
        emb = encode_text(query.text)
        print("🔢 Embedding computed.")

        pred = classifier_session.run(None, {"input": emb})[0]
        print("🔮 Prediction raw output:", pred)

        label_idx = int(pred[0])
        label = label_encoder.inverse_transform([label_idx])[0]
        print("✅ Final classification label:", label)
        return {"label": label}
    except Exception as e:
        print("❌ Error in classification:", str(e))
        return {"error": str(e)}

# === OpenAI GPT Streaming Endpoint ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/ask-stream")
async def ask_gpt_stream(query: QueryInput):
    try:
        def stream():
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful healthcare assistant of ASHA or ANM that replies in Hindi."},
                    {"role": "user", "content": query.text},
                ],
                stream=True,
            )
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content.encode("utf-8")
            yield b""  # Clean close

        return StreamingResponse(stream(), media_type="text/plain")
    except Exception as e:
        print("❌ Error in GPT stream:", str(e))
        return {"reply": f"⚠️ GPT streaming error: {str(e)}"}

# === OpenAPI Schema ===
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Swasthya Doot API",
        version="1.0",
        description="API for classifying Hindi queries and chatting with GPT in Hindi for ASHA/ANM support.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# === Health Check ===
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "🟢 API is live and healthy"}

# === Run on Render / Local ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port)