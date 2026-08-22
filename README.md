# Capestone Project
This repository contains a capstone project that unifies three AI components into one connected support system for Flipkart’s catalog and customer support teams:

1. **Return‑Risk Model**  – predicts whether an order is likely to be returned.

2. **Product Image Categoriser** – classifies catalog product photos using transfer learning.

3. **Support Agent (LangGraph)** – a single assistant that can:
   + Answer policy questions via retrieval‑augmented generation (RAG).
   + Call the trained return‑risk model.
   + Call the trained image classifier.

All parts are integrated into one end‑to‑end demo.

📂 Repository Structure
Code
```
Capstone-Project/
│
├── data/
│   └── sample_images/        # Exported Fashion-MNIST test images (.png)
│       ├── 01_trouser.png
│       ├── 02_pullover.png
│       ├── 04_coat.png
│       ├── 06_shirt.png
│       └── 09_ankle_boot.png
│
├── models/                   # Saved ML/DL model artifacts
│   ├── product_classifier.keras
│   ├── product_classifier.pt
│   ├── product_classifier.weights.h5
│   └── return_risk_model.pkl
│
├── transcripts/              # Agent demo conversations
│   ├── 01_policy_return_window.txt
│   ├── 02_policy_cod_refund.txt
│   ├── 03_return_risk_tool.txt
│   ├── 04_image_classifier_tool.txt
│   ├── 05_multiturn_state_carried.txt
│   ├── 06_fresh_conversation_reset.txt
│   ├── 07_prompt_injection_blocked.txt
│   └── 08_ungrounded_policy_refused.txt
│
├── flipkart_support_agent.py     # LangGraph agent (Part 3)
├── generate_orders.py            # Dataset generator (Part 1)
├── orders_dataset.csv            # Seeded dataset (6,000 rows)
├── product_image_categoriser.py  # Transfer learning classifier (Part 2)
├── return_risk_scoring.py        # Return-risk model training/evaluation
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

### ⚙️ Setup Instructions
Clone the repository and install dependencies:
```
git clone https://github.com/priyanshumore99/Capstone-Project.git
cd Capstone-Project
pip install -r requirements.txt
```

## 🧩 Part 1 – Return‑Risk Scoring Pipeline
Goal: Predict whether an order will be returned.

### Steps to Reproduce
1. **Generate dataset:**

```
python3 generate_orders.py
```
- Produces orders_dataset.csv with 6,000 rows × 13 columns.
* Deterministic output (seeded with np.random.default_rng(42)).
<br>

2. **Train and evaluate models:**
```
python3 return_risk_scoring.py
```
- Baseline: DummyClassifier (F1=0.0 for returned=1).
* Logistic Regression: ROC‑AUC ≥ 0.58, threshold sweep for F1.
+ Random Forest: GridSearchCV tuned, ROC‑AUC ≥ 0.58, saved as final artifact.
<br>

3. **Final artifact:**
- Saved pipeline: models/return_risk_model.pkl
* Threshold t*_rf recorded for risk buckets.
<br>

## 👗 Part 2 – Product Image Categoriser
**Goal:** Classify apparel/footwear/accessory images.
<br>

Run the below command to generate dataset.
```
python3 product_image_categoriser.py
```

### Steps to Reproduce
1. Dataset: **Fashion‑MNIST** (70,000 images, 10 classes).
- Train: 60,000 → split into train + validation.
* Test: 10,000 untouched until final evaluation.
<br>

2. Preprocessing:
- Grayscale → 3 channels.
* Resize to backbone input size (e.g., 224×224 for ResNet‑18).
+ Normalize with ImageNet mean/std.
<br>

3. Training:
- Backbone: ResNet‑18 (pretrained).
* Feature extraction → fine‑tuning if needed.
+ Optimizer: Adam, batch size documented.
<br>

4. Evaluation:
- Test accuracy ≥ 80%.
* Confusion matrix + per‑class precision/recall.
+ Confusion pairs explained (e.g., Shirt vs. T‑shirt).
<br>

5. Final artifact:
- Saved weights: models/product_classifier.pt
* Sample images exported to data/sample_images/.
<br>

## 🤖 Part 3 – Flipkart Support Agent (LangGraph)
**Goal:** One assistant that integrates policy KB + both trained models.
<br>

### Features
- **Policy KB:** 12+ documents chunked, covering return windows, COD refunds, SLAs, reverse pickup.
* **Vector Index:** Sentence‑transformers embeddings + Faiss/ChromaDB.
+ **Tools:**
  + check_return_risk(order_features: dict) → probability + bucket (Low/Medium/High, anchored to t*_rf).
  + classify_product_image(image_path: str) → predicted label + confidence.

- **LangGraph:**
  + Nodes: intent routing, RAG retrieval, tool calling, response generation.
  + Conditional branching by intent.
  + Conversational state maintained across turns.

- **MOCK_LLM:** Deterministic, zero API keys/network calls.

- **Guardrails:** Prompt‑injection filtering + groundedness check.

### Run Agent
```
python3 flipkart_support_agent.py
```
Runs in MOCK_LLM mode by default.
<br>
<br>

## 📜 Example Transcript
Located in transcripts/. Includes:
- Policy Q&A via RAG.
- Return‑risk prediction for an order.
- Product image classification from data/sample_images/.
- Multi‑turn conversation with state carried across turns.
- Fresh conversation showing state reset.
- Prompt‑injection attempt blocked.
- Ungrounded policy question refused with similarity score shown.
<br>

## 📊 Retrieval Evaluation
- Precision@3 and Recall@3 computed on 5+ queries.
- Per‑query arithmetic shown in transcripts.
- Average metrics reported in README.
