# src/geminapi.py (c) 2025 Gregory L. Magnusson MIT License

from typing import Optional, List, Dict, Any, Union, Generator
import google.generativeai as genai
# from google.generativeai.types import GenerationConfig  # No longer needed
import logging
from dataclasses import dataclass

# Initialize logger (make sure you have src.logger set up)
logger = logging.getLogger('geminapi')

@dataclass
class GeminiResponse:
    """Dataclass to structure Gemini API responses (simplified)."""
    response: str
    model: str
    # Add other fields as needed from the response object

class GeminiHandler:
    """Handles interactions with the Google Gemini API."""

    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("API key must be provided.")
        genai.configure(api_key=api_key)  # Configure the API key globally
        self.selected_model: Optional[str] = None
        self.temperature: float = 0.3
        self.streaming: bool = False

    def list_models(self, include_experimental: bool = False) -> List[str]:
        """Lists available Gemini models."""
        available_models = []
        try:
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    model_name = model.name
                    if include_experimental or "-exp" not in model_name:
                        available_models.append(model_name)
            return available_models
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            return []

    def select_model(self, model_name: str) -> bool:
        """Selects a Gemini model by its full name."""
        self.selected_model = model_name
        return True

    def set_temperature(self, temperature: float):
        """Sets the temperature."""
        if 0.0 <= temperature <= 1.0:
            self.temperature = temperature
        else:
            logger.warning("Temperature must be between 0.0 and 1.0.  Using default.")

    def set_streaming(self, streaming: bool):
        """Enables or disables streaming."""
        self.streaming = streaming

    def generate_response(self, prompt: str) -> Union[GeminiResponse, Generator[str, None, None], str]:
        """Generates a response."""
        if not self.selected_model:
            logger.error("No model selected.")
            return "Error: No Gemini model selected."

        try:
            model = genai.GenerativeModel(self.selected_model) # Use GenerativeModel
            if self.streaming:
                response_stream = model.generate_content(
                    prompt,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                    )
                )
                return self._stream_response(response_stream)

            else:
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                    )
                )
                return GeminiResponse(response=response.text, model=self.selected_model)


        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return f"Gemini Error: {e}"

    def _stream_response(self, response_stream: Any) -> Generator[str, None, None]:
        """Handles streaming responses."""
        for chunk in response_stream:
            yield chunk.text

    def get_last_error(self) -> Optional[str]:
        return None
