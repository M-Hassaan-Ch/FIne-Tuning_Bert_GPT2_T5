"""
Streamlit app for summarization using a fine-tuned T5 model (supports LoRA adapters).

Usage:
  1) Place your saved T5 model folder (the one you saved after training) somewhere accessible.
  2) If the folder contains a full seq2seq checkpoint, set `Model folder` to that path.
     If the folder contains only a LoRA adapter, set `Base model` in the sidebar to the base model name
     (for example: `t5-small`) and set `Model folder` to the adapter folder.
  3) Install dependencies:
       pip install streamlit transformers torch accelerate peft
     (choose the correct `torch` wheel for your system at https://pytorch.org/)
  4) Run:
       streamlit run e:/Uni/NLP/NLP_Assignment3/t5_app.py

Notes:
  - The app will attempt to load a seq2seq model from `Model folder`. If that fails and you provide
    a `Base model`, it will load the base model and attach the LoRA adapter using `peft` (if installed).
  - The UI exposes a single `Max target length` control (in the sidebar) to control summary length.
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Tuple, Any

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    PEFT_AVAILABLE = False


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_tokenizer_and_model(model_dir: str, base_model: str = None) -> Tuple[Any, Any, torch.device]:
    """Load tokenizer and T5 model. Try direct load first; if that fails and base_model
    is provided, try base+LoRA adapter via PEFT.
    """
    device = _get_device()

    # Try direct load from saved folder
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        if not base_model:
            raise

    # Fallback: load base model and attach LoRA adapter
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    base = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT not installed; install `peft` to load LoRA adapters (pip install peft)")

    model = PeftModel.from_pretrained(base, model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def summarize_text(tokenizer, model, device, text: str, max_target_length: int = 128):
    if not isinstance(text, str) or text.strip() == "":
        raise ValueError("Empty text provided")

    # Tokenize and move to device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_target_length, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    st.title("T5 Summarization — Streamlit")

    st.sidebar.header("Model settings")
    default_model_dir = "./t5_final_model"  # change after you add folder
    model_dir = st.sidebar.text_input("Model folder", value=default_model_dir, help="Path to your saved T5 model or adapter folder")
    base_model = st.sidebar.text_input("Base model (optional)", value="t5-small", help="If model folder is a LoRA adapter, set the base model name here (e.g., t5-small)")

    # Single length control in sidebar
    max_target_length = st.sidebar.number_input("Max target length", value=128, min_value=16, max_value=1024, help="Maximum tokens in generated summary")

    if st.sidebar.button("Load model"):
        st.experimental_rerun()

    tokenizer = model = device = None
    load_error = None
    try:
        tokenizer, model, device = load_tokenizer_and_model(model_dir, base_model if base_model.strip() else None)
        st.sidebar.success(f"Model loaded (device: {device})")
    except Exception as e:
        load_error = e
        st.sidebar.error(f"Model load failed: {e}")

    st.subheader("Input text to summarize")
    sample = st.text_area("Text", value="Type or paste the article/text here to summarize...", height=200)

    summarize_btn = st.button("Summarize")

    if summarize_btn:
        if model is None or tokenizer is None:
            st.error("Model not loaded. Set the model folder in the sidebar and click 'Load model'.")
        else:
            try:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(tokenizer, model, device, sample, max_target_length=int(max_target_length))
                st.subheader("Summary")
                st.write(summary)
                st.markdown("---")
                st.markdown("**Full output (raw decode)**")
                st.write(summary)
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.markdown("**Tips**: If you trained with LoRA and saved an adapter folder, set `Base model` to the original base (e.g., `t5-small`) and `Model folder` to the adapter folder. Click `Load model` after adding the folder.")


if __name__ == "__main__":
    main()
