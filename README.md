# Multimodal Text-to-Motion Retrieval Challenge
## 📌 Overview
This project focuses on a **multimodal retrieval problem** involving natural language descriptions and **3D human motion interaction sequences**.

The core objective is to learn a **shared representation space** (joint embedding) that aligns textual descriptions with their corresponding human motion sequences. 
Unlike generation or captioning, the focus here is strictly on **retrieval**: finding the correct motion given a text query, and vice-versa.

#### Key Focus: Human-Human Interactions
A significant portion of the challenge involves complex interactions such as:
* Handshaking
* Pushing
* Hugging
Understanding **roles, spatial relations, and temporal dynamics** is critical for success in this task.

## 🚀 Main Tasks
1. **Text → Motion Retrieval**
Given a natural language description (e.g., "A person gives a high-five to another"), the model must retrieve the most relevant motion sequence(s) from a candidate pool.

2. **Alignment Strategy**
The model should implement a **Contrastive Learning** approach where:
* Matching text–motion pairs are pulled together in the embedding space.
* Mismatched pairs are pushed apart.

## 📊 Evaluation Metrics
The performance is measured using standard cross-modal retrieval metrics, specifically **Recall@K**.

##### Recall@K
Recall@K measures how often the correct motion $m_i^*$ appears within the top $K$ results predicted by the similarity score $s(q_i, m_j)$.

##### Final Score Calculation
The leaderboard ranking is determined by a **weighted average of Recall@K** for $k \in \{1, 2, \dots, 10\}$:

$$Score = \sum_{k=1}^{10} \frac{Recall@k}{k}$$

*Note: Higher importance is given to lower ranks (e.g., Recall@1 is more valuable than Recall@10).*

## 💾 Submission Format
Submissions must be a CSV file containing the query_id and the top 10 ranked motion_ids.
 
| query_id | candidate_1 | candidate_2 | ... | candidate_10 |
|----------|-------------|-------------|-----|--------------|
| 1141     | 114         | 622         | ... | 237          | 
| 1142     | 892         | 532         | ... | 257          |

##### Constraints:
* Each candidate motion ID must be *unique* per query.
* Rankings must be ordered from *most relevant* (candidate_1) to *least relevant.* 

## 🧠 Solution
The reference implementation is [`notebook.ipynb`](notebook.ipynb) (best validation score: **0.9065** weighted Recall@K, epoch 30/60).

- **Text Encoder**: `all-mpnet-base-v2` (last 2 layers fine-tuned) → mean pooling → projection MLP
- **Motion Encoder**: Gated Person Fusion (per-frame weighting of the two persons) → BiGRU (2 layers, bidirectional) → attention pooling → projection MLP
- **Shared embedding**: 384 dimensions
- **Loss**: multi-positive InfoNCE + progressive hard negative mining + label smoothing
- **Training**: all 3 texts per motion as pairs, random crop, speed jitter (0.85–1.15×), person swap, Gaussian noise, EMA weights, top-3 checkpoint saving
- **Inference**: person-swap TTA, plus an optional rank blend with a previous submission

Mirror/flip augmentation is intentionally excluded: texts contain directional words ("to the left", "right hand").

### Repository layout
| Path | Content |
|------|---------|
| `notebook.ipynb` | Full pipeline: data, model, training, validation, submission (Kaggle or local) |
| `exploration.ipynb` | Data exploration, 3D motion visualization, evaluation baseline |
| `src/` | Earlier script-based version of the pipeline |
| `submission.csv` | Submission file |

## 📚 References & Research
This project is inspired by and references the following state-of-the-art work:
1. [TMR: Text-to-Motion Retrieval (Petrovich et al., ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Petrovich_TMR_Text-to-Motion_Retrieval_Using_Contrastive_3D_Human_Motion_Synthesis_ICCV_2023_paper.pdf)
2. [Multi-Instance Multi-Label Learning for Text-Motion Retrieval] (https://dl.acm.org/doi/pdf/10.1145/3664647.3681444)
3. [Modal-Enhanced Semantic Modeling for Fine-Grained 3D Human Motion Retrieval] (https://dl.acm.org/doi/pdf/10.1145/3664647.3681625)

## 🛠 Installation & Usage

```
# Clone the repository
git clone https://github.com/MissawB/Text-to-Motion.git

# Install dependencies
pip install -r requirements.txt
```

## 📝 Citation
```
@misc{tmr-text-motion-retrieval,
    author = {Hazem Wannous and IKEN OMAR},
    title = {TMR: Text-Motion Retrieval},
    year = {2026},
    publisher = {Kaggle},
    url = {https://kaggle.com/competitions/tmr-text-motion-retrieval}
}
```