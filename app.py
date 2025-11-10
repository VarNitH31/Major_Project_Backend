

# app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from inference import load_fused_embeddings, recommend_from_history
import pandas as pd

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data and embeddings
FUSED_PATH = "models/final_fusion_results.pt"
CSV_PATH = "models/products.csv"

df = pd.read_csv(CSV_PATH)
fused_embs, fusion_weights = load_fused_embeddings(FUSED_PATH)

@app.post("/recommend")
async def recommend(body: dict):
    indices = body.get("indices", [])
    top_k = body.get("top_k", 10)
    recs = recommend_from_history(indices, fused_embs, fusion_weights, df, top_k=top_k)

    # Merge recs with product data
    rec_products = [
        {**recs[i], **df.iloc[recs[i]["index"]].to_dict()}
        for i in range(len(recs))
        if recs[i]["index"] < len(df)
    ]
    return {"recommendations": rec_products}
