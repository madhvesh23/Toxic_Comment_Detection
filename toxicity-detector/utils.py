import torch
import shap
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ambiguous words and adversarial patterns
ambiguous_words = [
    "gay", "queer", "black", "white", "jew", "muslim", "islam", "christian", "female",
    "woman", "man", "asian", "indian", "african", "fat", "skinny", "sex", "gender",
    "trans", "homo", "straight", "race", "religion", "nationality"
]
adversarial_patterns = ["0", "1", "3", "@", "!", "$", "#", "*", "%", "^", "&"]

# Load both models once
model_clean = torch.load("checkpoints/best_model_1.pkl", map_location=device)
model_adv = torch.load("checkpoints/best_model_step7_2.pkl", map_location=device)

model_clean.to(device).eval()
model_adv.to(device).eval()

# Create SHAP pipelines
pipeline_clean = pipeline("text-classification", model=model_clean, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
pipeline_adv = pipeline("text-classification", model=model_adv, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

explainer_clean = shap.Explainer(pipeline_clean)
explainer_adv = shap.Explainer(pipeline_adv)

def sanitize_text(user_input):
    if not isinstance(user_input, str):
        raise ValueError(" Invalid input: expected a string.")
    cleaned = user_input.strip()
    if len(cleaned) == 0:
        raise ValueError(" Empty input after cleaning.")
    
    #  Apply normalization for adversarial text
    normalized = normalize_obfuscated_text(cleaned)
    return normalized

def contains_ambiguous(text):
    lowered = text.lower()
    return any(word in lowered for word in ambiguous_words)

def contains_adversarial(text):
    return any(char in text for char in adversarial_patterns)

def predict_and_explain(user_input):
    text = sanitize_text(user_input)
    input_list = [text]

    # Decide which model to use
    use_adv_model = contains_ambiguous(text) or contains_adversarial(text)
    model = model_adv if use_adv_model else model_clean
    explainer = explainer_adv if use_adv_model else explainer_clean

    # Prediction
    inputs = tokenizer(input_list, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        prediction = torch.argmax(probs).item()
        confidence = probs[prediction].item()

    # SHAP explanation
    shap_vals = explainer(input_list)
    explanation = shap_vals[0]

    # FP filter
    importances = explanation.values[:, 1] if explanation.values.ndim > 1 else explanation.values
    num_important = sum(abs(val) > 0.1 for val in importances)
    filter_flag = "Needs Review" if prediction == 1 and num_important == 1 and confidence < 0.7 else "Okay"

    # Flags
    ambiguous_flag = contains_ambiguous(text)
    adversarial_flag = contains_adversarial(text)

    # Top contributors
    tokens = explanation.data
    top_contributors = sorted(zip(tokens, importances), key=lambda x: abs(x[1]), reverse=True)[:5]
    top_tokens = [f"{tok} ({val:+.4f})" for tok, val in top_contributors]

    return {
        "model_used": "BERT + Robust (Step7_2)" if use_adv_model else "BERT Clean Only",
        "prediction": "Toxic" if prediction == 1 else "Non-Toxic",
        "confidence": round(confidence * 100, 2),
        "shap": explanation,
        "filter_flag": filter_flag,
        "ambiguous_flag": ambiguous_flag,
        "adversarial_flag": adversarial_flag,
        "top_contributors": top_tokens
    }



import unicodedata

def normalize_obfuscated_text(text):
    """
    Normalize stylized unicode characters to ASCII equivalent.
    """
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')