import os
import random
import csv
import numpy as np
import pandas as pd
from os.path import join as pjoin

def generate_val_batches(data_root, save_dir, batch_size=32, num_batches=5):
    """Génère un mini-dataset de validation à partir du train pour calculer le score."""
    print(f"Génération du validation set dans {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(data_root, 'train.txt')) as fd:
        train_fnames = fd.read().strip().split('\n') 

    # On prend un échantillon aléatoire
    random_fnames = random.sample(train_fnames, k=batch_size * num_batches)
    random_batches = np.array(random_fnames).reshape(num_batches, batch_size)

    gt = [['query_id', 'candidate']]
    
    # Nettoyage préalable si le dossier existe déjà (optionnel mais conseillé)
    # Ici on écrase simplement
    
    for file_idx, batch in enumerate(random_batches, start=1):
        batch_dir = pjoin(save_dir, str(file_idx))
        os.makedirs(batch_dir, exist_ok=True)

        rdm_idx = random.randint(0, len(batch)-1)
        random_text_fname = batch[rdm_idx]

        # Sauvegarde du texte (Query)
        with open(os.path.join(data_root, 'texts', random_text_fname+'.txt')) as fd:
            file_texts = fd.read().strip().split('\n')
            random_text = random.choice(file_texts)

        with open(os.path.join(batch_dir, 'text.txt'), 'w') as fw:
            fw.write(random_text)

        # Sauvegarde des mouvements (Candidats)
        for motion_idx, fname in enumerate(batch, start=1):
            if fname == random_text_fname:
                gt.append([file_idx, motion_idx])

            motion = np.load(os.path.join(data_root, 'motions', fname+'.npy'))
            np.save(pjoin(batch_dir, 'motion_{}.npy'.format(motion_idx)), motion)

    # Sauvegarde du fichier Ground Truth
    with open(os.path.join(save_dir, 'gt.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(gt)
        
    print('✅ Validation batches générés !')

def eval_recall(gt_df, pred_df, verbose=False):
    ks = list(range(1, 11))  # Recall@1 ... Recall@9
    rank_cols = list(pred_df.columns)

    # Convert to numpy
    gt = gt_df["candidate"].values
    preds = pred_df[rank_cols].values

    assert len(gt) == len(preds), "size mismatch"

    # ---- Compute Recall@K ----
    recalls = {k: 0 for k in ks}
    n = len(gt)

    for i in range(n):
        for k in ks:
            if gt[i] in preds[i, :k]:
                recalls[k] += 1

    for k in ks:
        recalls[k] /= n

    # ---- Weighted aggregation ----
    weights = {k: 1.0 / k for k in ks}
    weight_sum = sum(weights.values())

    final_score = sum(weights[k] * recalls[k] for k in ks) / weight_sum
    
    if verbose:
        print(f"Score Weighted Recall: {final_score:.4f}")
        
    return final_score