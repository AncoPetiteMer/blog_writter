import os
from dotenv import load_dotenv

class APIConfig:
    def __init__(self, dotenv_path: str = ".env"):
        load_dotenv(dotenv_path)
        self.OPENAI_API_KEY = self._get_key("OPENAI_API_KEY")
        self.TAVILY_API_KEY = self._get_key("TAVILY_API_KEY")
        self.EXA_API_KEY = self._get_key("EXA_API_KEY")
        self.SEMRUSH_API_KEY = self._get_key("SEMRUSH_API_KEY")

    def _get_key(self, key_name: str) -> str:
        value = os.getenv(key_name)
        if not value:
            raise EnvironmentError(f"❌ La variable d'environnement '{key_name}' est manquante.")
        return value

