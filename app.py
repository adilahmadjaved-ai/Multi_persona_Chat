import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Multi-Persona Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Persona AI Chatbot")

# -----------------------------
# LOAD MODEL
# -----------------------------
MODEL_NAME = "microsoft/phi-2"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_model()

# -----------------------------
# PERSONAS
# -----------------------------
PERSONAS = {
    "Coder":
        "You are a senior software engineer. "
        "Answer programming questions clearly with code examples.",

    "Therapist":
        "You are a calm, supportive therapist. "
        "Respond empathetically and respectfully.",

    "Sales":
        "You are a professional salesperson. "
        "Recommend products persuasively but honestly."
}

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Settings")

persona = st.sidebar.selectbox(
    "Choose Persona",
    list(PERSONAS.keys())
)

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "system",
            "content": PERSONAS[persona]
        }
    ]

# Update system prompt when persona changes
if st.session_state.messages[0]["content"] != PERSONAS[persona]:
    st.session_state.messages = [
        {
            "role": "system",
            "content": PERSONAS[persona]
        }
    ]

# -----------------------------
# DISPLAY CHAT
# -----------------------------
for msg in st.session_state.messages:

    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# USER INPUT
# -----------------------------
prompt = st.chat_input("Type your message...")

if prompt:

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Keep only last 5 conversation turns
    history = st.session_state.messages[1:]

    if len(history) > 10:
        history = history[-10:]

    conversation = st.session_state.messages[:1] + history

    # Build prompt manually
    full_prompt = ""

    for msg in conversation:

        if msg["role"] == "system":
            full_prompt += f"System: {msg['content']}\n"

        elif msg["role"] == "user":
            full_prompt += f"User: {msg['content']}\n"

        elif msg["role"] == "assistant":
            full_prompt += f"Assistant: {msg['content']}\n"

    full_prompt += "Assistant:"

    inputs = tokenizer(
        full_prompt,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with st.spinner("Thinking..."):

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    assistant_reply = response.split("Assistant:")[-1].strip()

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_reply
        }
    )