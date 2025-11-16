"""
Streamlit app to serve predictions from your fine-tuned BERT model.

How to use:
  1) Place your fine-tuned model folder (for example: "bert-sentiment-final") in a
     location on this machine.
  2) Start this app with Streamlit:
       pip install streamlit transformers torch
       streamlit run e:/Uni/NLP/NLP_Assignment3/bert_app.py

  3) In the app sidebar set `Model folder` to the path of your saved model.

Notes:
  - The app uses `transformers.AutoTokenizer` and
    `transformers.AutoModelForSequenceClassification` — this matches the
    architecture you used in the notebook.
  - If the model directory contains a `config.json` with `id2label`, that mapping
    will be used. Otherwise we fall back to {0: "NEGATIVE", 1: "POSITIVE"}.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

# Default fallback mapping (used when model.config.id2label isn't present)
# Enforce lowercase labels so label 0 -> 'negative' and label 1 -> 'positive'
DEFAULT_ID2LABEL = {0: "negative", 1: "positive"}


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_and_tokenizer(model_dir: str) -> Tuple[Any, Any, Dict[int, str], torch.device]:
    """Load tokenizer and model from `model_dir`. Cached by Streamlit.

    Note: When you change `model_dir` in the sidebar Streamlit will re-run and
    reload the model (the cache key includes the argument).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = _get_device()
    model.to(device)
    model.eval()

    # If the model is binary (num_labels == 2) enforce the mapping:
    #   0 -> 'negative'
    #   1 -> 'positive'
    # Otherwise, prefer any mapping provided in model.config.id2label,
    # and finally fall back to DEFAULT_ID2LABEL.
    if getattr(model.config, "num_labels", None) == 2:
        id2label = {0: "negative", 1: "positive"}
    else:
        id2label = getattr(model.config, "id2label", None)
        if id2label is None:
            id2label = DEFAULT_ID2LABEL

    return tokenizer, model, id2label, device


def predict_text(tokenizer, model, id2label, device: torch.device, text: str, max_length: int = 128):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Empty text provided")

    inputs = tokenizer(text, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(probs.argmax())
        score = float(probs[pred_id])

    label = id2label.get(pred_id, f"LABEL_{pred_id}")

    return {"label": label, "label_id": pred_id, "score": score, "probs": probs.tolist()}


def main():
    st.title("BERT Sentiment — Streamlit")

    st.sidebar.header("Model settings")
    default_model_dir = "./bert-sentiment-final"  # change after you add folder
    model_dir = st.sidebar.text_input("Model folder", value=default_model_dir, help="Path to your saved model folder (set this after adding the folder)")
    st.sidebar.markdown("\nExample: `./bert-sentiment-final`")

    # Show a model load button so user intentionally attempts to load.
    if st.sidebar.button("Load model"):
        st.experimental_rerun()

    st.write("\n")

    # Show loading status / try to load model
    tokenizer = model = id2label = device = None
    load_error = None
    try:
        tokenizer, model, id2label, device = load_model_and_tokenizer(model_dir)
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        load_error = e
        st.sidebar.error(f"Model failed to load: {e}")

    st.subheader("Enter text to analyze")
    sample = st.text_area("Text", value="I love this product!", height=120)

    col1, col2 = st.columns([1, 3])
    with col1:
        max_len = st.number_input("Max tokens", min_value=16, max_value=512, value=128)
        predict_btn = st.button("Predict")

    if predict_btn:
        if model is None or tokenizer is None:
            st.error("Model not loaded. Set the model folder in the sidebar and click 'Load model'.")
        else:
            try:
                out = predict_text(tokenizer, model, id2label, device, sample, max_length=int(max_len))
                st.success(f"Prediction: {out['label']} (score: {out['score']:.4f})")
                st.write("Label id:", out["label_id"])
                st.write("Probabilities:")
                # show probs as simple table
                for i, p in enumerate(out["probs"]):
                    lbl = id2label.get(i, f"LABEL_{i}")
                    st.write(f"- {lbl}: {p:.4f}")
                # bar chart
                try:
                    import pandas as pd

                    df = pd.DataFrame([out["probs"]], columns=[id2label.get(i, f"LABEL_{i}") for i in range(len(out["probs"]))])
                    st.bar_chart(df.T)
                except Exception:
                    pass
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.markdown("**Tips**: Add your model folder, then click `Load model` in the sidebar. If you save a new model with the same path, reload the page to refresh cached state.")


if __name__ == "__main__":
    main()
