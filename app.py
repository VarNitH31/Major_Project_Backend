# # app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# import json
# import torch
# from inference import load_model, get_fused_embedding, load_fused_embeddings, recommend_from_history
# from fastapi.middleware.cors import CORSMiddleware

# MODEL_PATH = "best_fusion_model.pt"
# FUSED_PATH = "final_fusion_results.pt"
# PRODUCTS_PATH = "products.json"
# EMBED_DIM = 256

# app = FastAPI(title="Hierarchical Fusion Recommender API")


# # Allow frontend origin(s)
# origins = [
#     "http://localhost:3000",  # your Next.js frontend
#     "http://127.0.0.1:3000",
#     # You can add your deployed domain here later, e.g.:
#     # "https://your-frontend.vercel.app"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,        # allowed domains
#     allow_credentials=True,
#     allow_methods=["*"],          # allow all HTTP methods (GET, POST, etc.)
#     allow_headers=["*"],          # allow all headers
# )

# # Load model and precomputed embeddings
# model = load_model(MODEL_PATH, EMBED_DIM)
# fused_embs, fusion_weights = load_fused_embeddings(FUSED_PATH)



# # Load product metadata
# with open(PRODUCTS_PATH, "r", encoding="utf-8") as f:
#     products = json.load(f)


# class EmbeddingRequest(BaseModel):
#     text_embeddings: list
#     image_embeddings: list


# class HistoryRequest(BaseModel):
#     indices: list[int]
#     top_k: int = 10


# @app.post("/fuse")
# def fuse_embeddings(req: EmbeddingRequest):
#     fused_emb, fusion_weights_ = get_fused_embedding(model, req.text_embeddings, req.image_embeddings)
#     return {
#         "fused_embedding": fused_emb,
#         "fusion_weights": fusion_weights_
#     }


# @app.post("/recommend")
# def recommend_products(req: HistoryRequest):
#     print("history indices:", req.indices)
#     recs = recommend_from_history(req.indices, fused_embs, fusion_weights, top_k=req.top_k)
#     # attach product info
#     print("product 896:", products[896])
#     for i in range(len(recs)):
#         print("recommended product:", products[recs[i]["index"]])
#     rec_products = [
#         {**recs[i], **products[recs[i]["index"]]} for i in range(len(recs)) if recs[i]["index"] < len(products)
#     ]
#     return {"recommendations": rec_products}

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
