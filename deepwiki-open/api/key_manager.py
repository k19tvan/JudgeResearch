import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GoogleKeyManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GoogleKeyManager, cls).__new__(cls)
            cls._instance.keys = cls._instance._load_keys()
            cls._instance.current_index = 0
            if cls._instance.keys:
                cls._instance.configure_genai()
        return cls._instance

    def _load_keys(self) -> list:
        keys_str = os.getenv("GOOGLE_API_KEYS")
        if keys_str:
            return [k.strip() for k in keys_str.split(",") if k.strip()]
        
        # Fallback to single key
        key = os.getenv("GOOGLE_API_KEY")
        if key:
            return [key.strip()]
            
        return []

    def configure_genai(self):
        if not self.keys:
            return
        
        current_key = self.keys[self.current_index]
        genai.configure(api_key=current_key)
        
    def rotate_key(self):
        if not self.keys:
            return
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.warning(f"Google API Key rotated. Now using key index {self.current_index}.")
        self.configure_genai()

key_manager = GoogleKeyManager()
