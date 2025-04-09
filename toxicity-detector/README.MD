# Toxic Comment Detection with BERT + SHAP + Adversarial Robustness

![Python](https://img.shields.io/badge/python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-💬-brightgreen)
![BERT](https://img.shields.io/badge/BERT-transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A powerful and explainable toxic comment classification system built using:
- **BERT** for deep contextual understanding of text
- **SHAP** for interpretability and explanation visualization
- **Adversarial Training + False Positive Filtering** for robustness

> ✨ Supports real-time inference via Streamlit web app.

---

## Table of Contents

- [🚀 Features](#-features)
- [🧪 Model Architecture](#-model-architecture)
- [📦 Setup Instructions](#-setup-instructions)
- [💻 Running the App](#-running-the-app)
- [📊 Evaluation & Results](#-evaluation--results)
- [🧠 SHAP Visualization](#-shap-visualization)
- [📁 Project Structure](#-project-structure)
- [📌 Future Work](#-future-work)
- [🤝 Contributors](#-contributors)

---

## Features

Fine-tuned BERT-based binary classifier  
Real-time toxicity prediction with confidence score  
SHAP explanations for interpretability  
Ambiguous word & adversarial pattern detection  
False Positive Filter with token-level logic  
Dual-model routing (clean vs adversarial robust)  
Streamlit UI for interactive demo

---

## Model Architecture

- Base Model: `bert-base-uncased`
- Fine-tuned on toxic vs non-toxic comments
- Adversarial fine-tuning: obfuscated inputs
- False Positive Filtering Logic:
  - 1 important token + confidence < 70% → `Needs Review`

---

## Setup Instructions

```bash
# Clone repository
git clone https://github.com/madhvesh23/Toxic_Comment_Detection.git
cd toxic-detector

# Create virtual environment
conda create -n toxicenv python=3.9
conda activate toxicenv

# Install dependencies
pip install -r requirements.txt

# Running the app
# Start Streamlit app
streamlit run app.py