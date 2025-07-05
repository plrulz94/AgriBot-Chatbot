# ✅ STEP 1: Install required libraries
!pip install transformers sentencepiece streamlit pyngrok --quiet

# ✅ STEP 2: Kill any old ngrok tunnels
from pyngrok import ngrok
ngrok.kill()

# ✅ STEP 3: Save Streamlit chatbot app (FLAN-T5 based)
app_code = """
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="AgriBot (FLAN-T5)", layout="centered")
st.title("🌾 AgriBot - Smart Farming Assistant (FLAN-T5)")
st.markdown("Ask me anything about agriculture, irrigation, soil, fertilizer, or crops.")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

tokenizer, model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []

for past_input, past_output in st.session_state.history:
    st.chat_message("user").markdown(past_input)
    st.chat_message("assistant").markdown(past_output)

user_input = st.chat_input("👨‍🌾 Ask your question...")
if user_input:
    st.chat_message("user").markdown(user_input)

    prompt = f"Answer the following agricultural question clearly:\\n{user_input}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append((user_input, answer))
"""

with open("app.py", "w") as f:
    f.write(app_code)

# ✅ STEP 4: Add ngrok authtoken
!ngrok config add-authtoken 3pBRBt5hQxQbZRRgDTfyg_4u5eUiKKnTjed7K6CU4mN

# ✅ STEP 5: Launch Streamlit and expose via ngrok
import subprocess, time
!pkill streamlit
process = subprocess.Popen(["streamlit", "run", "app.py"])
time.sleep(15)  # Wait for Streamlit to start

public_url = ngrok.connect(8501)
print("🌐 AgriBot FLAN-T5 is live! Click below:")
print(public_url)
