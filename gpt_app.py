

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Tuple, Any

try:
    # peft is optional; used if user wants to load LoRA adapters
    from peft import PeftModel
    PEFT_AVAILABLE = True
except Exception:
    PeftModel = None
    PEFT_AVAILABLE = False


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_tokenizer_and_model(model_dir: str, base_model: str = None):
    """Load tokenizer and model.

    - If `base_model` is provided and loading `model_dir` directly fails, the function
      will try to load the base model and then attach the LoRA adapter from `model_dir`
      using `PeftModel.from_pretrained` (if `peft` is installed).
    - Returns (tokenizer, model, device)
    """
    device = _get_device()

    # Try to load model directly from `model_dir` (works if it's a complete model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception:
        # Fall back to base_model + adapter if provided
        if not base_model:
            raise

    # If we reach here, attempt base model + LoRA adapter
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(base_model)

    if not PEFT_AVAILABLE:
        raise RuntimeError("PEFT is not installed; install `peft` to load LoRA adapters (pip install peft)")

    model = PeftModel.from_pretrained(base, model_dir)
    model.to(device)
    model.eval()

    return tokenizer, model, device


def generate_text(tokenizer, model, device, prompt: str, max_new_tokens: int = 128):
    """Simple generation using only `max_new_tokens` for length control.

    Other sampling parameters are left at model/default settings.
    """
    if not isinstance(prompt, str) or prompt.strip() == "":
        raise ValueError("Empty prompt provided")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") else None

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def main():
    st.title("GPT Generation — Streamlit")

    st.sidebar.header("Model settings")
    default_model_dir = "./gpt2-pseudo-code-translator-final"  # change after you add folder
    model_dir = st.sidebar.text_input("Model folder", value=default_model_dir, help="Path to your saved model folder (adapter or full model)")
    base_model = st.sidebar.text_input("Base model (optional)", value="", help="If model folder is a LoRA adapter, set the base model name here (e.g., openai-community/gpt2)")
    st.sidebar.markdown("If you provide a base model the app will try to attach LoRA adapter from `Model folder`.")

    if st.sidebar.button("Load model"):
        st.experimental_rerun()

    # Place the length control in the sidebar under model settings (left column)
    max_new_tokens = st.sidebar.number_input("Max new tokens", value=128, min_value=1, max_value=2048, help="Maximum number of tokens to generate")

    tokenizer = model = device = None
    load_error = None
    try:
        tokenizer, model, device = load_tokenizer_and_model(model_dir, base_model if base_model.strip() else None)
        st.sidebar.success(f"Loaded model (device: {device})")
    except Exception as e:
        load_error = e
        st.sidebar.error(f"Model load failed: {e}")

    st.subheader("Generation prompt")
    prompt = st.text_area("Prompt", value="Translate the following pseudo-code to working code:\nPseudo-code: print hello world\nCode:", height=200)

    # Generate button remains in main column
    generate_btn = st.button("Generate")

    if generate_btn:
        if model is None or tokenizer is None:
            st.error("Model not loaded. Set the model folder and (optionally) base model, then click 'Load model'.")
        else:
            try:
                with st.spinner("Generating..."):
                    text = generate_text(tokenizer, model, device, prompt, max_new_tokens=int(max_new_tokens))
                # Show result and also the continuation (strip prompt if prompt present at start)
                if text.startswith(prompt):
                    continuation = text[len(prompt):]
                else:
                    continuation = text
                st.subheader("Generated continuation")
                st.code(continuation)
                st.markdown("---")
                st.markdown("**Full output (prompt + generation)**")
                st.write(text)
            except Exception as e:
                st.exception(e)

    st.markdown("---")
    st.markdown("**Tips**: If you used LoRA during training, set the `Base model` to the original base (for example `openai-community/gpt2`) and set the `Model folder` to the adapter folder.")


if __name__ == "__main__":
    main()
