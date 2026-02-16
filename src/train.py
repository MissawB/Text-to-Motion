import os
import glob
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
# import pandas as pd
from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import time

# Imports locaux
from model import Config, MotionTextDataset, collate_fn, DualEncoder, contrastive_loss
# from utils import generate_val_batches, eval_recall

# --- FONCTION DE VALIDATION (INTERNE) ---
# def validate(model, tokenizer, val_dir):
#     """Calcule le score sur le dossier de validation généré."""
#     model.eval()
#     submission_rows = []
    
#     # Lister les dossiers batchs (1, 2, 3...)
#     batch_folders = sorted([f for f in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, f)) and f.isdigit()], key=lambda x: int(x))
    
#     with torch.no_grad():
#         for batch_id in batch_folders:
#             batch_path = os.path.join(val_dir, batch_id)
            
#             # 1. Lire la Query
#             with open(os.path.join(batch_path, 'text.txt'), 'r') as f:
#                 query_text = f.read().strip()
                
#             tokens = tokenizer([query_text], padding=True, truncation=True, max_length=128, return_tensors='pt').to(Config.DEVICE)
#             text_feat = model.text_encoder(tokens['input_ids'], tokens['attention_mask'])
#             text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
#             # 2. Lire les Candidats
#             motion_files = glob.glob(os.path.join(batch_path, '*.npy'))
#             candidate_motions = []
#             candidate_ids = []
            
#             for m_file in motion_files:
#                 # Format attendu: motion_{id}.npy
#                 base_name = os.path.splitext(os.path.basename(m_file))[0] # motion_1
#                 c_id = int(base_name.split('_')[-1]) # 1
#                 candidate_ids.append(c_id)
                
#                 motion = np.load(m_file)
#                 if motion.shape[0] > Config.MAX_LEN:
#                     motion = motion[:Config.MAX_LEN, :]
#                 candidate_motions.append(torch.from_numpy(motion).float())
            
#             # Padding & Forward
#             motions_padded = pad_sequence(candidate_motions, batch_first=True, padding_value=0.0).to(Config.DEVICE)
#             motion_feats = model.motion_encoder(motions_padded)
#             motion_feats = motion_feats / motion_feats.norm(dim=-1, keepdim=True)
            
#             # 3. Score
#             scores = torch.matmul(text_feat, motion_feats.T).squeeze(0)
#             sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
            
#             # Top 10 IDs
#             top_10_ids = [candidate_ids[i] for i in sorted_indices[:10]]
#             submission_rows.append([int(batch_id)] + top_10_ids)

#     # Création du DataFrame Preds
#     cols = ['query_id'] + [f'candidate_{i}' for i in range(1, 11)]
#     pred_df = pd.DataFrame(submission_rows, columns=cols).set_index('query_id').sort_index()
    
#     # Chargement du GT
#     gt_df = pd.read_csv(os.path.join(val_dir, 'gt.csv'))
    
#     # Calcul du score
#     score = eval_recall(gt_df, pred_df, verbose=False)
#     return score

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. Générer les données de validation (Validation Set)
    val_dir = os.path.join(Config.DATA_ROOT, 'val_generated')
    # On génère 10 batchs de 32 exemples pour avoir une estimation fiable
    # generate_val_batches(Config.DATA_ROOT, val_dir, batch_size=32, num_batches=10)

    # 2. Setup Data & Model
    dataset = MotionTextDataset(Config.DATA_ROOT, split_file='train.txt')
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = DualEncoder(Config).to(Config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)

    # Historiques pour le plot
    history = {'loss': [], 
            #    'score': []
            }

    print(f"Dataset size: {len(dataset)}")
    print("Starting Training Loop...")

    start_time = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        
        # --- TRAIN LOOP ---
        for batch_idx, (motions, texts, lengths) in enumerate(dataloader):
            motions = motions.to(Config.DEVICE)
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt').to(Config.DEVICE)
            
            optimizer.zero_grad()
            t_emb, m_emb = model(tokens['input_ids'], tokens['attention_mask'], motions)
            loss = contrastive_loss(t_emb, m_emb, model.temperature)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        # --- VALIDATION LOOP ---
        print(f"Epoch {epoch+1} ended. Running Validation...")
        # current_score = validate(model, tokenizer, val_dir)
        
        # --- LOGGING & PLOTTING ---
        history['loss'].append(avg_loss)
        # history['score'].append(current_score)
        
        print(f"📊 Epoch {epoch+1} | Loss: {avg_loss:.4f}")
        print(f"⏰ Time elapsed: {time.time() - start_time:.2f}s")
        # Mise à jour du graphique en temps réel (sauvegardé dans un fichier)
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        # plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch+2), history['loss'], 'b-o', label='Train Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Evolution')
        plt.grid(True)
        
        # # Plot Score
        # plt.subplot(1, 2, 2)
        # plt.plot(range(1, epoch+2), history['score'], 'g-o', label='Val Recall Score')
        # plt.xlabel('Epochs')
        # plt.ylabel('Weighted Recall')
        # plt.title('Validation Score Evolution')
        # plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close() # Fermer pour libérer la mémoire

    # Sauvegarde finale
    torch.save(model.state_dict(), "dual_encoder_model.pth")
    print("✅ Training Done! Model saved.")