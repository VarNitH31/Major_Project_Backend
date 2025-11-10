
# inference.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Model definition
# ===============================
class HierarchicalFusionMultiHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.text_proj = nn.ModuleList([nn.Linear(embed_dim, hidden_dim) for _ in range(num_heads)])
        self.image_proj = nn.ModuleList([nn.Linear(embed_dim, hidden_dim) for _ in range(num_heads)])
        self.text_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        self.image_gate = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_heads)])
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, text_emb, image_emb):
        head_fused, head_weights = [], []
        for h in range(self.num_heads):
            t_h = F.relu(self.text_proj[h](text_emb))
            i_h = F.relu(self.image_proj[h](image_emb))
            t_w = torch.sigmoid(self.text_gate[h](t_h))
            i_w = torch.sigmoid(self.image_gate[h](i_h))
            w = torch.cat([t_w, i_w], dim=-1)
            w = F.softmax(w, dim=-1)
            head_weights.append(w.unsqueeze(1))
            fused_h = torch.cat([t_h*w[:,0:1], i_h*w[:,1:2]], dim=-1)
            head_fused.append(fused_h.unsqueeze(1))
        fused_multi = torch.mean(torch.cat(head_fused, dim=1), dim=1)
        fused_final = F.normalize(self.fusion(fused_multi), p=2, dim=1)
        fusion_weights = torch.mean(torch.cat(head_weights, dim=1), dim=1)
        return fused_final, fusion_weights

# ===============================
# Load fused embeddings
# ===============================
def load_fused_embeddings(path):
    fused_embs, fusion_weights = torch.load(path, map_location=DEVICE)
    fused_embs = F.normalize(fused_embs, p=2, dim=1)
    return fused_embs, fusion_weights

# ===============================
# Load model weights (optional)
# ===============================
def load_fusion_model(weight_path, embed_dim=512):
    model = HierarchicalFusionMultiHead(embed_dim)
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    return model

# ===============================
# Recommend from user history
# ===============================
def recommend_from_history(indices, fused_embs, fusion_weights, df, top_k=5):
    indices = torch.tensor(indices, dtype=torch.long, device=DEVICE)
    history_embs = fused_embs[indices]
    decay = 0.84
    weights = torch.tensor([decay**(len(indices) - 1 - i) for i in range(len(indices))], device=DEVICE)
    weights = weights / weights.sum()

    profile = (history_embs * weights[:, None]).sum(dim=0)
    profile = F.normalize(profile, p=2, dim=0)

    sims = util.cos_sim(profile.unsqueeze(0), fused_embs)[0]
    sims[indices] = -1  # exclude history items
    top_scores, top_idx = torch.topk(sims, k=top_k)

    recs = []
    for score, idx in zip(top_scores, top_idx):
        recs.append({
            "index": idx.item(),
            "score": score.item(),
            "text_weight": fusion_weights[idx][0].item(),
            "image_weight": fusion_weights[idx][1].item(),
        })
    return recs
