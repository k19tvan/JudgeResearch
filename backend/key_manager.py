import os
import logging
from google import genai

logger = logging.getLogger(__name__)

class BackendGoogleKeyManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendGoogleKeyManager, cls).__new__(cls)
            cls._instance.keys = cls._instance._load_keys()
            cls._instance.current_index = 0
            cls._instance.client = None
            if cls._instance.keys:
                cls._instance._configure_client()
        return cls._instance

    def _load_keys(self) -> list:
        keys_str = os.getenv("GEMINI_API_KEYS")
        if keys_str:
            return [k.strip() for k in keys_str.split(",") if k.strip()]
        
        # Fallback to single key
        key = os.getenv("GEMINI_API_KEY")
        if key:
            return [key.strip()]
            
        return []

    def _configure_client(self):
        if not self.keys:
            return
        
        current_key = self.keys[self.current_index]
        self.client = genai.Client(api_key=current_key)
        
    def rotate_key(self):
        if not self.keys:
            return
        
        self.current_index = (self.current_index + 1) % len(self.keys)
        logger.warning(f"Backend API Key rotated. Now using key index {self.current_index}.")
        self._configure_client()
        
    def get_client(self):
        return self.client

key_manager = BackendGoogleKeyManager()
