import os
import warnings
import logging

# Silence warnings and block network pings for instant loading
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PIL import Image
import timm
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import hf_hub_download
from io import BytesIO

app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

#ARCHITECTURE DEFINITIONS
class ConvStem(nn.Module):
    def __init__(self):
        super(ConvStem, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)

class TransformerMIL(nn.Module):
    def __init__(self, input_dim=1536, embed_dim=128, num_heads=4, num_classes=3):
        super(TransformerMIL, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.ReLU(), nn.Dropout(0.6)
        )
        self.attention_V = nn.Sequential(nn.Linear(embed_dim, 64), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(embed_dim, 64), nn.Sigmoid())
        self.attention_weights = nn.Linear(64, 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*2, 
            dropout=0.6, activation='relu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        h = self.feature_extractor(x)
        z = self.transformer(h)
        A = F.softmax(self.attention_weights(self.attention_V(z) * self.attention_U(z)).squeeze(2) / 0.5, dim=1)
        bag_rep = torch.bmm(A.unsqueeze(1), z).squeeze(1)
        return self.classifier(bag_rep), A

#GLOBAL VRAM LOADING (Runs Once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing FastAPI Server on: {device}")

print("Locking CTransPath & Phikon into VRAM...")
ct_m = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)
ct_m.patch_embed = ConvStem()
weights = hf_hub_download(repo_id="jamesdolezal/CTransPath", filename="ctranspath.pth", local_files_only=True)
ct_m.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
ct_m = ct_m.to(device).half().eval()

p_proc = AutoImageProcessor.from_pretrained("owkin/phikon", local_files_only=True)
p_m = AutoModel.from_pretrained("owkin/phikon", local_files_only=True).to(device).half().eval()

print("Locking 5-Fold Ensemble into VRAM...")
ensemble = []
for i in range(5):
    m = TransformerMIL().to(device)
    m.load_state_dict(torch.load(f"./ensemble/brain_{i}.pth", map_location=device))
    m.eval()
    ensemble.append(m)

print("VRAM Setup Complete. Server Ready.")

# Define the endpoints
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    classes = ['Benign', 'Malignant', 'Normal']
    
    # 1. Read Image
    contents = await file.read()
    img_pil = Image.open(BytesIO(contents)).convert('RGB')
    w, h = img_pil.size

    # 2. Extract Patches (Fast CPU Gathering)
    patch_size, stride = 256, 128
    ct_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    bag_feats, patches_pil = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_pil.crop((x, y, x + patch_size, y + patch_size))
            patches_pil.append(patch)

    # 3. GPU Batch Processing
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(patches_pil), batch_size):
            batch_patches = patches_pil[i : i + batch_size]
            
            p_in = p_proc(images=batch_patches, return_tensors="pt").to(device)
            p_in['pixel_values'] = p_in['pixel_values'].half()
            p_out = p_m(**p_in).last_hidden_state[:, 0, :]
            
            c_in = torch.stack([ct_transform(p) for p in batch_patches]).to(device).half()
            c_out = ct_m(c_in)
            
            batch_feats = torch.cat((p_out, c_out), dim=1)
            bag_feats.append(batch_feats)

    bag_tensor = torch.cat(bag_feats, dim=0).unsqueeze(0).float()

    # 4. 5-Fold MIL Ensemble Prediction
    all_probs = []
    with torch.no_grad():
        for model in ensemble:
            logits, _ = model(bag_tensor)
            all_probs.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(all_probs).mean(dim=0).squeeze().cpu().numpy()
    pred_idx = np.argmax(avg_probs)
    score_breakdown = {classes[i]: round(float(avg_probs[i] * 100), 2) for i in range(len(classes))}

    # Clean unused memory
    del bag_tensor
    torch.cuda.empty_cache()

    # 5. Return JSON Package
    return {
        "filename": file.filename,
        "prediction": classes[pred_idx],
        "all_scores": score_breakdown
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)