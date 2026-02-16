import os
import csv
import glob
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

# Import de ton architecture existante
from model import Config, DualEncoder

def generate_submission(model_weights_path="dual_encoder_model.pth", submission_file="submission.csv"):
    
    # 1. Configuration
    device = Config.DEVICE
    print(f"⚙️  Génération sur : {device}")
    
    # 2. Recharger le modèle
    # On crée une instance vide et on charge les poids
    model = DualEncoder(Config).to(device)
    
    if not os.path.exists(model_weights_path):
        print(f"❌ Erreur : Le fichier de poids {model_weights_path} est introuvable.")
        return

    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval() # DÉSACTIVE le Dropout (Crucial pour des résultats stables)
    print("✅ Modèle chargé.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 3. Préparation
    test_root = os.path.join(Config.DATA_ROOT, 'test')
    submission_rows = []
    
    # Récupérer les dossiers 1, 2, ... 30 dans le bon ordre
    batch_folders = [f for f in os.listdir(test_root) if not f.startswith('.')]
    # Tri numérique (sinon 10 arrive avant 2)
    batch_folders = sorted(batch_folders, key=lambda x: int(x))
    
    print(f"🚀 Traitement de {len(batch_folders)} requêtes de test...")

    with torch.no_grad(): # Pas de gradients = plus rapide et moins de mémoire
        for batch_id in batch_folders:
            batch_path = os.path.join(test_root, batch_id)
            
            # --- A. Query (Texte) ---
            query_path = os.path.join(batch_path, 'text.txt')
            with open(query_path, 'r') as f:
                query_text = f.read().strip()
                
            # Tokenization
            tokens = tokenizer([query_text], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            
            # Encodage Texte
            text_feat = model.text_encoder(tokens['input_ids'], tokens['attention_mask'])
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True) # Normalisation
            
            # --- B. Candidats (Mouvements) ---
            motion_files = glob.glob(os.path.join(batch_path, '*.npy'))
            candidate_motions = []
            candidate_ids = []
            
            for m_file in motion_files:
                # Récupération de l'ID. Ex: "motion_12.npy" -> 12
                base_name = os.path.splitext(os.path.basename(m_file))[0]
                try:
                    c_id = int(base_name.split('_')[-1])
                except ValueError:
                    c_id = base_name # Fallback si le format change
                
                candidate_ids.append(c_id)
                
                # Chargement
                motion = np.load(m_file)
                # Truncate si trop long (comme à l'entrainement)
                if motion.shape[0] > Config.MAX_LEN:
                    motion = motion[:Config.MAX_LEN, :]
                candidate_motions.append(torch.from_numpy(motion).float())
            
            # Padding
            motions_padded = pad_sequence(candidate_motions, batch_first=True, padding_value=0.0).to(device)
            
            # Encodage Mouvement
            motion_feats = model.motion_encoder(motions_padded)
            motion_feats = motion_feats / motion_feats.norm(dim=-1, keepdim=True) # Normalisation
            
            # --- C. Similarité & Ranking ---
            # Matrice de similarité (1, 32)
            scores = torch.matmul(text_feat, motion_feats.T).squeeze(0)
            
            # On trie les indices par score décroissant
            sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
            
            # On garde le TOP 10
            top_10_indices = sorted_indices[:10]
            top_10_ids = [candidate_ids[i] for i in top_10_indices]
            
            # On ajoute la ligne pour le CSV
            submission_rows.append([batch_id] + top_10_ids)

    # 4. Écriture du CSV
    header = ['query_id'] + [f'candidate_{i}' for i in range(1, 11)]
    
    with open(submission_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(submission_rows)
        
    print(f"🎉 Fichier généré : {submission_file}")

if __name__ == "__main__":
    generate_submission()