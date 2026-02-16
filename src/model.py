import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel

# --- CONFIGURATION ---
class Config:
    DATA_ROOT = './data'      # Ton chemin actuel
    BATCH_SIZE = 32
    LR = 1e-4
    EPOCHS = 20               # Commence petit pour tester
    MAX_LEN = 200             # On coupe les mouvements trop longs
    EMBED_DIM = 256           # Dimension de l'espace latent partagé
    MOTION_INPUT_DIM = 384    # Confirmé par tes soins
    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

print(f"🚀 Training on: {Config.DEVICE}")

# --- 1. DATASET ---
class MotionTextDataset(Dataset):
    def __init__(self, data_root, split_file='train.txt', max_len=200):
        self.data_root = data_root
        self.max_len = max_len
        
        split_path = os.path.join(data_root, split_file)
        with open(split_path, 'r') as f:
            self.ids = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        motion_id = self.ids[idx]
        
        # Load Motion
        motion_path = os.path.join(self.data_root, 'motions', f"{motion_id}.npy")
        motion = np.load(motion_path)
        
        # Truncate
        if motion.shape[0] > self.max_len:
            motion = motion[:self.max_len, :]
        
        motion_tensor = torch.from_numpy(motion).float()
        
        # Load Text
        text_path = os.path.join(self.data_root, 'texts', f"{motion_id}.txt")
        with open(text_path, 'r') as f:
            # On sait qu'il y a 3 lignes, on en prend une au hasard
            texts = [line.strip() for line in f.readlines() if line.strip()]
            selected_text = np.random.choice(texts)
            
        return motion_tensor, selected_text

def collate_fn(batch):
    motions, texts = zip(*batch)
    # Padding: (Batch, Max_T, 384)
    motions_padded = pad_sequence(motions, batch_first=True, padding_value=0.0)
    # Création du masque d'attention pour le GRU (optionnel mais propre)
    lengths = torch.tensor([len(m) for m in motions])
    return motions_padded, list(texts), lengths

# --- 2. MODÈLES ---
class TextEncoder(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", embed_dim=256):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # On peut geler les 2 premières couches pour stabiliser le début
        # for param in self.bert.embeddings.parameters():
        #     param.requires_grad = False
            
        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim)
        )

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # On utilise le token [CLS]
        return self.projection(output.last_hidden_state[:, 0, :])

class MotionEncoder(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, embed_dim=256):
        super().__init__()
        
        # Petite astuce : Conv1D pour mélanger les features spatiales avant le temporel
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512), # *2 car Bidirectionnel
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, motions):
        # motions: (B, T, 384) -> (B, T, 512)
        x = self.feature_extractor(motions)
        
        # GRU
        output, _ = self.gru(x)
        
        # Mean Pooling sur l'axe temporel (dim 1)
        # On pourrait utiliser les longueurs réelles pour faire une moyenne plus précise
        # mais la moyenne simple marche souvent bien si le padding est à 0
        mean_motion = output.mean(dim=1)
        
        return self.projection(mean_motion)

class DualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(embed_dim=cfg.EMBED_DIM)
        self.motion_encoder = MotionEncoder(input_dim=cfg.MOTION_INPUT_DIM, embed_dim=cfg.EMBED_DIM)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, input_ids, attention_mask, motions):
        text_feat = self.text_encoder(input_ids, attention_mask)
        motion_feat = self.motion_encoder(motions)
        
        # Normalisation L2 (Essentiel pour InfoNCE)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        motion_feat = motion_feat / motion_feat.norm(dim=-1, keepdim=True)
        
        return text_feat, motion_feat

# --- 3. LOSS FUNCTION (InfoNCE) ---
def contrastive_loss(text_feat, motion_feat, temperature):
    # Similitude Cosinus (déjà normalisé) -> (Batch, Batch)
    logits = torch.matmul(text_feat, motion_feat.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    
    loss_t2m = nn.CrossEntropyLoss()(logits, labels)
    loss_m2t = nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_t2m + loss_m2t) / 2