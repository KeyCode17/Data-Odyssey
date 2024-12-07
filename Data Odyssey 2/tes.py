
import os
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Streamlit page configuration
st.set_page_config(page_title="AI Chat Assistant", page_icon="💬", layout="centered")

# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define the path where the model is saved
MODEL_SAVE_DIR = './model_artifacts/fine_tuned_model'

# Load model and tokenizer with caching
@st.cache_resource
def load_model():
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SAVE_DIR)
    
    # Set the pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set device to CPU
    device = torch.device('cpu')
    
    # Load the model
    model = GPT2LMHeadModel.from_pretrained(MODEL_SAVE_DIR).to(device)
    
    return model, tokenizer, device

# Load model, tokenizer, and device
model, tokenizer, device = load_model()

def generate_text(prompt="The", max_length=50):  
    # Encode the input prompt  
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)  

    # Generate text  
    with torch.no_grad():  
        output = model.generate(  
            input_ids,   
            max_length=max_length,   
            num_return_sequences=1,  
            no_repeat_ngram_size=2,  
            do_sample=True,  
            top_k=50,  
            top_p=0.95,  
            pad_token_id=tokenizer.eos_token_id  
        )  

    # Decode and return the generated text  
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Custom CSS for chat styling
st.markdown("""
<style>
.user-message {
    background-color: #e6f3ff;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
    max-width: 80%;
    align-self: flex-end;
    text-align: right;
    margin-left: auto;
}
.bot-message {
    background-color: #f0f0f0;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
    max-width: 80%;
    align-self: flex-start;
    text-align: left;
    margin-right: auto;
}
.chat-container {
    display: flex;
    flex-direction: column;
    height: 500px;
    overflow-y: auto;
    padding: 20px;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 AI Chat Assistant")
st.write("Chat with your fine-tuned AI model!")

# Chat container
st.markdown('<div class="chat-container" id="chat-container">', unsafe_allow_html=True)

# Display previous messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">{message["content"]}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.text_input("Your message:", key="user_input")

# Send button
if st.button("Send"):
    if user_input:
        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate bot response
        try:
            bot_response = generate_text(prompt=user_input, max_length=100)
            
            # Add bot message to session state
            st.session_state.messages.append({"role": "bot", "content": bot_response})
        except Exception as e:
            st.error(f"Error generating response: {e}")
        
        # Rerun to update the chat display
        st.rerun()

# Optional: Clear chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Model information display
st.sidebar.title("Model Information")
st.sidebar.write(f"Model Path: {MODEL_SAVE_DIR}")
st.sidebar.write(f"Device: {device}")

