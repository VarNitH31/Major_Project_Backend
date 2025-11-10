# fusion_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(hidden_dim*2, hidden_dim),
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
