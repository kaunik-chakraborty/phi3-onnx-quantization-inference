import streamlit as st
import os
from pathlib import Path
from run_quantized_model import QuantizedModelRunner

# Initialize the model runner
@st.cache_resource
def get_model_runner():
    return QuantizedModelRunner()

runner = get_model_runner()

st.set_page_config(layout="wide", page_title="Quantized Multi-Model Chatbot")
st.title("🤖 Quantized Multi-Model Chatbot")

st.markdown("""
<style>
    .st-emotion-cache-10o4u5r {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-z5fcl4 {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .st-emotion-cache-1y4pz8r {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-1c7y2km {
        background-color: #e6f7ff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-1r4qj8v {
        font-size: 1.1em;
    }
</style>
""", unsafe_allow_html=True)

# Model Selection
models = runner.discover_quantized_models()
model_names = list(models.keys())

if not model_names:
    st.error("No quantized models found! Please run quantization first.")
else:
    st.sidebar.header("Model Configuration")
    selected_model_name = st.sidebar.selectbox(
        "Select a model:",
        model_names,
        index=model_names.index(st.session_state.get('selected_model', model_names[0])) if st.session_state.get('selected_model') else 0,
        key="model_selector"
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Developed by Reformation AI")
    st.session_state['selected_model'] = selected_model_name

    # Load model if not already loaded or if a different model is selected
    if runner.current_model_name != selected_model_name:
        with st.spinner(f"Loading {selected_model_name}..."):
            model_path = models[selected_model_name]
            if runner.load_model(model_path, selected_model_name):
                st.success(f"Loaded {selected_model_name} successfully!")
                # Clear chat history when a new model is loaded
                st.session_state.messages = []
            else:
                st.error(f"Failed to load {selected_model_name}.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare input for model
            conversation_history = []
            for msg in st.session_state.messages:
                conversation_history.append({"role": msg["role"], "content": msg["content"]})

            input_ids = runner.current_tokenizer.apply_chat_template(
                conversation_history,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            # Generate response
            try:
                outputs = runner.current_model.generate(
                    input_ids,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=runner.current_tokenizer.eos_token_id
                )
                
                # Decode response
                full_response = runner.current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new response
                prompt_text = runner.current_tokenizer.apply_chat_template(
                    conversation_history,
                    tokenize=False,
                    add_generation_prompt=True
                )
                response_text = full_response[len(prompt_text):].strip()
                message_placeholder.markdown(response_text)
            except Exception as e:
                st.error(f"Error during inference: {e}")
                response_text = f"Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": response_text})