from fastapi import FastAPI, HTTPException, Security, status, Header
from pydantic import BaseModel, Field
from openai import OpenAI
import uvicorn
import os
from dotenv import load_dotenv
try:
    import onnxruntime as ort
except ImportError:
    print("⚠️ ONNX Runtime not found. Classifier features will be disabled.")
    ort = None
import numpy as np
from transformers import AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from huggingface_hub import hf_hub_download
from fastapi.openapi.utils import get_openapi
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import PreTrainedTokenizerFast
from typing import Optional, List

# === Imports for ABDM ===
# Try/Except to avoid breaking the app if routes are missing during dev
try:
    from routes.abdm_routes import router as abdm_router
except ImportError as e:
    print(f"⚠️ ABDM Routes Import Failed: {e}")
    abdm_router = None

# === Init ===
load_dotenv()
app = FastAPI()

# === ABDM Router Registration ===
if abdm_router:
    app.include_router(abdm_router)

# === Config ===
API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

if not API_KEY:
    print("⚠️ WARNING: API_KEY not set in .env. API is unsecured!")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Security ===
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return x_api_key

# === Globals ===
encoder_session = None
tokenizer = None

# === Constants ===
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "classifier.onnx")
PIPELINE_PATH = os.path.join(os.path.dirname(__file__), "classifier_pipeline_light.pkl")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer")
HUB_REPO_ID = "panghal/swasthya-encoder"
HUB_FILENAME = "encoder_quantized.onnx"

# === Load classifier pipeline ===
try:
    if ort:
        print("📦 Loading classifier pipeline...")
        scaler, label_encoder, _ = joblib.load(PIPELINE_PATH)
        classifier_session = ort.InferenceSession(CLASSIFIER_PATH)
    else:
        raise ImportError("ONNX Runtime missing")
except Exception as e:
    print(f"⚠️ Classifier Load Warning: {e}")
    scaler = None
    label_encoder = None
    classifier_session = None

# === Schema ===
class QueryInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

# === Lazy encoder loader ===
def get_encoder():
    global encoder_session, tokenizer

    if encoder_session is None and ort:
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
    if not encoder:
        return None
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=128)
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    outputs = encoder.run(None, ort_inputs)
    token_embeddings = outputs[0]
    return scaler.transform(token_embeddings.astype(np.float32))

# === Classifier Endpoint ===
@app.post("/classify", dependencies=[Security(verify_api_key)])
def classify_query(query: QueryInput):
    try:
        if classifier_session is None:
             return {"label": "Unknown (Model not loaded)"}
        emb = encode_text(query.text)
        if emb is None:
             return {"label": "Unknown (Encoder error)"}

        pred = classifier_session.run(None, {"input": emb})[0]

        label_idx = int(pred[0])
        label = label_encoder.inverse_transform([label_idx])[0]
        return {"label": label}
    except Exception as e:
        print("❌ Error in classification:", str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal Server Error"}
        )

# === OpenAI GPT Streaming Endpoint ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/ask-stream", dependencies=[Security(verify_api_key)])
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

@app.get("/onnx-test", dependencies=[Security(verify_api_key)])
def onnx_test():
    try:
        if ort:
            sess = ort.InferenceSession(CLASSIFIER_PATH)
            return {"status": "ONNX model loaded successfully"}
        return {"status": "ONNX Runtime not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/tokenizer-test", dependencies=[Security(verify_api_key)])
def tokenizer_test():
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(TOKENIZER_PATH, "tokenizer.json"))
        return {"status": "Tokenizer loaded successfully"}
    except Exception as e:
        return {"error": str(e)}



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
