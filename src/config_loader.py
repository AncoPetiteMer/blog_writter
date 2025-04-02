import json
from pydantic import BaseModel, ValidationError


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.7


class TavilyConfig(BaseModel):
    enabled: bool = True
    search_depth: str = "moderate"
    max_results: int = 3


class SemrushConfig(BaseModel):
    display_limit: int = 5
    database: str = "us"


class WordCountConfig(BaseModel):
    min: int = 300
    max: int = 500


class BlogConfig(BaseModel):
    language: str = "french"
    word_count: WordCountConfig
    html_format: bool = True


class Config(BaseModel):
    llm: LLMConfig
    tavily: TavilyConfig
    semrush: SemrushConfig
    blog: BlogConfig


def load_config(path: str = "config.json") -> Config:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return Config(**raw)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(f"❌ Failed to load config: {e}")
