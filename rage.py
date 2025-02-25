# rage.py (c) 2025 Gregory L. Magnusson MIT License

import sys
import json
import time
import psutil
import streamlit as st
from pathlib import Path
from typing import Optional, Union, Generator, List, Dict, Any
from src.logger import get_logger
from src.openmind import OpenMind
from src.locallama import OllamaHandler, OllamaResponse
from src.geminapi import GeminiHandler, GeminiResponse
from src.memory import (
    memory_manager,
    ContextEntry,
    store_conversation,
    ContextType
)
import PIL.Image
import io
import google.generativeai as genai
from dotenv import load_dotenv, set_key, find_dotenv  # Import dotenv functions
import os

# Set the favicon and page title
st.set_page_config(
    page_title="RAGE",
    page_icon="gfx/rage.ico",
    layout="wide"
)

logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generative Engine"""

    def __init__(self):
        self.setup_session_state()
        self.load_css()
        self.memory = memory_manager
        self.openmind = OpenMind()
        self.chat_history = []  # Initialize chat history

    def setup_session_state(self):
        """Initialize session state variables."""
        session_vars = {
            "messages": [],
            "provider": "Ollama",
            "selected_model": None,
            "model_instances": {'ollama': None, 'gemini': None},
            "process_running": False,
            "show_search": False,
            "temperature": 0.30,
            "streaming": False,
            "current_response": "",
            "include_experimental": False,
            "gemini_api_key": "",
            "gemini_api_key_valid": False,
            "uploaded_file": None,
            "system_prompt": "",
        }
        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

        # Load API key from .env if it exists, and validate
        load_dotenv()
        if "GOOGLE_GENAI_API_KEY" in os.environ:
            api_key = os.environ["GOOGLE_GENAI_API_KEY"]
            if self.validate_gemini_api_key(api_key):
                st.session_state.gemini_api_key = api_key
                st.session_state.gemini_api_key_valid = True


    def load_css(self):
        """Load external CSS."""
        try:
            with open("gfx/styles.css", "r") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Could not find 'gfx/styles.css'.")

    def display_diagnostics(self):
        """Display system diagnostics."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        st.markdown(
            f'<div class="diagnostics-box">CPU: {cpu}% | RAM: {ram}%</div>',
            unsafe_allow_html=True
        )

    def setup_sidebar(self):
        """Configure sidebar."""
        with st.sidebar:
            st.markdown("### Configuration")

            if st.session_state.provider == "Gemini":
                st.markdown("#### Gemini API Key")
                api_key_input = st.text_input(
                    "Enter your Gemini API Key",
                    type="password",
                    key="gemini_api_key_input",
                    label_visibility="collapsed",
                    value=st.session_state.gemini_api_key  # Pre-fill with existing key
                )

                if api_key_input:
                    if self.validate_gemini_api_key(api_key_input):
                        st.session_state.gemini_api_key = api_key_input
                        st.session_state.gemini_api_key_valid = True
                        st.markdown("✅ API Key Accepted")

                        # Save to .env
                        dotenv_path = find_dotenv()  # Find the .env file
                        set_key(dotenv_path, "GOOGLE_GENAI_API_KEY", api_key_input)

                    else:
                        st.session_state.gemini_api_key_valid = False
                        st.error("Invalid Gemini API Key")

            if st.session_state.provider == "Gemini" and not st.session_state.gemini_api_key_valid:
                st.markdown("#### Select Provider")
                st.session_state.provider = st.selectbox(
                "Provider",
                options=["Ollama", "Gemini"],
                index=0 if st.session_state.provider == "Ollama" else 1,
                label_visibility="collapsed"
            )
            else:
                st.markdown("#### Select Provider")
                st.session_state.provider = st.selectbox(
                    "Provider",
                    options=["Ollama", "Gemini"],
                    index=0 if st.session_state.provider == "Ollama" else 1,
                    label_visibility="collapsed"
                )

                st.markdown("#### Select Model")
                if st.session_state.provider == "Ollama":
                    ollama_running, models = self.check_ollama_status()
                    if ollama_running and models:
                        st.session_state.selected_model = st.selectbox(
                            "Model Selection",
                            options=models,
                            index=0 if models else None,
                            label_visibility="collapsed"
                        )
                elif st.session_state.provider == "Gemini":
                    st.session_state.include_experimental = st.toggle(
                        "Include Experimental Models",
                        value=False
                    )
                    gemini_models = self.get_gemini_models()
                    if gemini_models:
                        st.session_state.selected_model = st.selectbox(
                            "Model Selection",
                            options=gemini_models,
                            index=0 if gemini_models else None,
                            label_visibility="collapsed",
                            format_func=lambda x: x.replace("models/", "")
                        )
                    else:
                        st.warning("No Gemini models available.")

                st.markdown("#### Temperature")
                st.session_state.temperature = st.slider(
                    "Temperature Control",
                    min_value=0.00,
                    max_value=1.00,
                    value=st.session_state.temperature,
                    step=0.01,
                    label_visibility="collapsed",
                    format="%.2f"
                )

                st.session_state.streaming = st.toggle(
                    "Enable Streaming",
                    value=st.session_state.streaming,
                    key="streaming_toggle"
                )
                st.markdown("#### System Prompt (Gemini Only)")
                st.session_state.system_prompt = st.text_area(
                    "Enter System Prompt (Instructions for the AI)",
                    value=st.session_state.system_prompt,
                    help="Provide instructions for the AI's behavior.",
                    label_visibility="collapsed",
                    height=150
                )

                if st.session_state.messages:
                    st.markdown("### Conversation History")
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            truncated_content = msg["content"][:50]
                            if len(msg["content"]) > 50:
                                truncated_content += "..."
                            st.markdown(
                                f'<div class="chat-history-item">{truncated_content}</div>',
                                unsafe_allow_html=True
                            )

    def validate_gemini_api_key(self, api_key: str) -> bool:
        """Validates the Gemini API key."""
        try:
            genai.configure(api_key=api_key)
            genai.list_models()
            return True
        except Exception as e:
            logger.error(f"Gemini API key validation failed: {e}")
            return False

    def check_ollama_status(self):
        """Check Ollama installation and available models."""
        try:
            if not st.session_state.model_instances['ollama']:
                st.session_state.model_instances['ollama'] = OllamaHandler()

            if st.session_state.model_instances['ollama'].check_installation():
                models = st.session_state.model_instances['ollama'].list_models()
                return True, models
            return False, []
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False, []

    def get_gemini_models(self):
        """Get available Gemini models."""
        try:
            if not st.session_state.model_instances['gemini']:
                if st.session_state.gemini_api_key:
                     st.session_state.model_instances['gemini'] = GeminiHandler(api_key=st.session_state.gemini_api_key)
                else:
                    return []
            return st.session_state.model_instances['gemini'].list_models(
                include_experimental=st.session_state.include_experimental
            )
        except Exception as e:
            logger.error(f"Error getting Gemini models: {e}")
            return []

    def initialize_model(self):
        """Initialize model instance (Ollama or Gemini)."""
        provider = st.session_state.provider
        if provider == "Ollama":
            return self.initialize_ollama()
        elif provider == "Gemini":
            return self.initialize_gemini()
        else:
            st.error(f"Unknown provider: {provider}")
            return None

    def initialize_ollama(self) -> Optional[OllamaHandler]:
        """Initialize Ollama instance."""
        try:
            if not st.session_state.model_instances['ollama']:
                st.session_state.model_instances['ollama'] = OllamaHandler()

            if st.session_state.model_instances['ollama'].check_installation():
                available_models = st.session_state.model_instances['ollama'].list_models()
                if available_models:
                    if not st.session_state.selected_model:
                        st.info("Please select an Ollama model")
                        return None

                    if st.session_state.model_instances['ollama'].select_model(st.session_state.selected_model):
                        return st.session_state.model_instances['ollama']
                    else:
                        st.error(st.session_state.model_instances['ollama'].get_last_error())
                        return None
                else:
                    st.error("No Ollama models found.")
                    return None
            else:
                st.error("Ollama service is not running.")
                return None
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            st.error(f"Error initializing Ollama: {str(e)}")
            return None

    def initialize_gemini(self) -> Optional[GeminiHandler]:
        """Initialize Gemini instance."""
        try:
            if not st.session_state.model_instances['gemini']:
                if st.session_state.gemini_api_key:
                    st.session_state.model_instances['gemini'] = GeminiHandler(api_key=st.session_state.gemini_api_key)
                else:
                    st.info("Please enter your Gemini API Key.")
                    return None

            if st.session_state.selected_model:
                st.session_state.model_instances['gemini'].select_model(st.session_state.selected_model)
                return st.session_state.model_instances['gemini']
            else:
                st.info("Please select a Gemini model")
                return None

        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            st.error(f"Error initializing Gemini: {str(e)}")
            return None


    def prepare_gemini_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Prepares the prompt for Gemini."""
        contents = []

        if st.session_state.provider == "Gemini" and st.session_state.system_prompt:
            contents.append({"role": "system", "parts": [{"text": st.session_state.system_prompt}]})

        for message in self.chat_history:
            contents.append({"role": message["role"], "parts": [{"text": message["content"]}]})

        if st.session_state.uploaded_file is not None:
            try:
                image = PIL.Image.open(st.session_state.uploaded_file)
                contents.append({"role": "user", "parts": [image, {"text": prompt}]})

            except Exception as e:
                st.error(f"Error processing image: {e}")
                return []
        else:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

        return contents


    def process_message(self, prompt: str):
        """Process user input."""
        if not prompt and st.session_state.uploaded_file is None:
            return

        try:
            model = self.initialize_model()
            if not model:
                return

            model.set_temperature(st.session_state.temperature)
            model.set_streaming(st.session_state.streaming)

            self.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                if st.session_state.uploaded_file:
                    st.image(st.session_state.uploaded_file, width=200)
                st.markdown(prompt)


            with st.chat_message("assistant"):
                message_placeholder = st.empty()

                if st.session_state.provider == "Gemini":
                    full_prompt = self.prepare_gemini_prompt(prompt)
                else:
                    context = self.memory.get_relevant_context(prompt)
                    user_prompt = self.openmind.get_user_prompt().format(
                        query=prompt,
                        context=context
                    )
                    full_prompt = f"{self.openmind.get_system_prompt()}\n\n{user_prompt}"


                start_time = time.time()

                if st.session_state.streaming:
                    full_response = ""
                    response_generator = model.generate_response(full_prompt)

                    if isinstance(response_generator, Generator):
                        for chunk in response_generator:
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                        message_placeholder.markdown(full_response)

                    elif isinstance(response_generator, str):
                        message_placeholder.markdown(response_generator)
                        full_response = response_generator
                    else:
                        message_placeholder.markdown("An unexpected streaming error occurred.")
                        full_response = "An unexpected streaming error occurred."
                    response_text = full_response
                else:
                    with st.spinner("RAGE is thinking..."):
                        response = model.generate_response(full_prompt)

                    if isinstance(response, OllamaResponse):
                        response_text = response.response
                    elif isinstance(response, GeminiResponse):
                        response_text = response.response
                    elif isinstance(response, str):
                        response_text = response
                    else:
                        response_text = "An unexpected error occurred."
                    message_placeholder.markdown(response_text)

                elapsed_time = time.time() - start_time
                message_placeholder.markdown(f"{response_text}\n\n*Response time: {elapsed_time:.2f}s*")

                if response_text:
                    self.chat_history.append({"role": "model", "content": response_text})

                    if st.session_state.provider == "Ollama":
                        store_conversation(ContextEntry(
                            content=f"Q: {prompt}\nA: {response_text}",
                            context_type=ContextType.CONVERSATION,
                            source="user",
                            metadata={
                                "provider": st.session_state.provider,
                                "model": st.session_state.selected_model,
                                "context": context if st.session_state.provider == "Ollama" else ""
                            }
                        ))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{response_text}\n\n*Response time: {elapsed_time:.2f}s*"
                    })

        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"Processing error: {str(e)}")

    def run(self):
        """Main application flow."""
        self.display_diagnostics()
        self.setup_sidebar()

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        st.session_state.uploaded_file = st.file_uploader(
            "Upload an image (optional)",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )

        prompt = st.chat_input(
            placeholder="DeepSeek with RAGE...",
            key="chat_input"
        )
        st.markdown(
            """
            <div class="button-group">
                <button class="stButton" title="Upload files">📁</button>
                <button class="stButton" title="Stop process">⏹️</button>
                <button class="stButton" title="Search">🔍</button>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if prompt or st.session_state.uploaded_file:
            self.process_message(prompt)

def main():
    RAGE().run()

if __name__ == "__main__":
    main()
