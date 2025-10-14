# 🤖 Repo_AmazonReview-RLHF

**Amazon Product Review Summarization** project built with  
**Supervised Fine-Tuning (SFT)** → **Direct Preference Optimization (DPO)** → **Reinforcement Learning from Human Feedback (RLHF)**.

---

## 🧩 Project Overview

This project explores how to generate high-quality summaries from Amazon product reviews.  
Starting from a baseline summarization model (BART / LLaMA-2), it progressively improves alignment with human preference through DPO and RLHF.

---

## 🚀 Workflow

| Phase | Objective | Input / Output | Core Tool |
|-------|------------|----------------|-----------|
| **0. Data Preparation** | Load & clean `sentence-transformers/amazon-reviews` dataset | `{review, summary}` | 🤖 `datasets`, `pandas` |
| **1. SFT (Supervised Fine-Tuning)** | Train baseline summarizer (`review → title`) | `Trainer` + ROUGE eval | 🧠 `transformers` |
| **2. DPO (Preference Alignment)** | Fine-tune using preference pairs `{prompt, chosen, rejected}` | Align model behavior | ⚖️ `trl.DPOTrainer` |
| **3. RLHF (Reward Optimization)** | Train with reward scores via PPO | `{query, response, reward}` | 🎮 `trl.PPOTrainer` |
| **4. Evaluation & Demo** | Human evaluation + Gradio demo | Generated summaries | 💬 `gradio` |

---

## 📚 Dataset

> [`sentence-transformers/amazon-reviews`](https://huggingface.co/datasets/sentence-transformers/amazon-reviews)

| Field | Description |
|-------|--------------|
| `review` | Full text of the product review |
| `title`  | Short user-written title used as summary (training label) |

---

## 🧠 Model Stack

- **Base Models:** `facebook/bart-base`, `google/pegasus-small`, or `meta-llama/Llama-2-7b-chat`
- **Training:** `HuggingFace Transformers` + `TRL` + `Accelerate`
- **Quantization:** `BitsAndBytes (4-bit / 8-bit)`
- **Evaluation:** `ROUGE`, `BERTScore`
- **Demo:** `Gradio` / `Streamlit`

---

## 🗂️ Folder Structure
Repo_AmazonReview-RLHF/
├── data/
│ ├── raw/ # Original HuggingFace data
│ ├── processed/ # Cleaned CSVs (review-summary pairs)
│ └── samples/ # Mini subsets for quick tests
├── notebooks/
│ ├── phase0_data_preparation.ipynb
│ ├── phase1_sft_training.ipynb
│ ├── phase2_dpo_training.ipynb
│ ├── phase3_rlhf_training.ipynb
│ └── phase4_evaluation_demo.ipynb
├── scripts/
│ ├── data_cleaning.py
│ ├── train_sft.py
│ ├── train_dpo.py
│ ├── train_rlhf.py
│ └── evaluate.py
├── reports/
│ ├── evaluation.md
│ ├── summary_examples.md
│ └── results/
├── requirements.txt
└── README.md
