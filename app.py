import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Multi-Persona Chatbot", page_icon="🤖")

st.title("🤖 Multi-Persona AI Chatbot")

# -------------------------
# MODEL
# -------------------------

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# -------------------------
# PERSONAS
# -------------------------

personas = {
    "Coder":
        "You are an expert software engineer. Answer with code examples whenever possible.",

    "Therapist":
        "You are a calm, supportive therapist who gives empathetic advice.",

    "Sales":
        "You are a professional sales assistant who recommends products politely."
}

persona = st.sidebar.selectbox(
    "Choose Persona",
    list(personas.keys())
)

# -------------------------
# MEMORY
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask something...")

if prompt:

    st.session_state.history.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    history = st.session_state.history[-10:]

    messages = [
        {
            "role": "system",
            "content": personas[persona]
        }
    ]

    messages.extend(history)

    formatted_prompt = ""

    for msg in messages:

        formatted_prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"

    formatted_prompt += "Assistant:"

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    assistant_reply = response.split("Assistant:")[-1].strip()

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )

    st.rerun()
