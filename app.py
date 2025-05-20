import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Function for top-p sampling
def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_mask = cumulative_probs > p
    if torch.any(sorted_mask):
        sorted_mask[torch.where(sorted_mask)[0][0] + 1:] = True
    sorted_logits[sorted_mask] = float('-inf')

    probs = torch.softmax(sorted_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    return sorted_indices[sampled_index], sorted_logits, probs, sorted_indices

# Load model and tokenizer once
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Top-p Sampling (Nucleus Sampling) with GPT-2")
prompt = st.text_input("Enter your prompt:", "The future of AI is")
p = st.slider("Top-p value", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
top_k = st.slider("Top-k tokens to visualize", min_value=5, max_value=50, value=20)

if st.button("Generate Next Token"):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

    sampled_token, filtered_logits, probs, sorted_indices = top_p_sampling(logits, p=p)
    sampled_word = tokenizer.decode(sampled_token)

    st.markdown(f"### Sampled next token: **'{sampled_word}'**")

    # Prepare data for visualization
    top_tokens = sorted_indices[:top_k]
    top_probs = probs[:top_k].detach().numpy()
    top_words = [tokenizer.decode([idx]) for idx in top_tokens]

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_words, top_probs)
    ax.set_title(f"Top-{top_k} Tokens Probability (Top-p={p})")
    ax.set_ylabel("Probability")
    plt.xticks(rotation=45)
    st.pyplot(fig)
