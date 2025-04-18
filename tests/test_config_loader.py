import os
import json
import pytest
import tempfile
from src.config_loader import load_config, Config, LLMConfig, TavilyConfig, SemrushConfig, BlogConfig, WordCountConfig

class TestConfigLoader:
    def test_load_config_valid_file(self):
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            config_data = {
                "llm": {
                    "model": "test-model",
                    "temperature": 0.5
                },
                "tavily": {
                    "enabled": True,
                    "search_depth": "basic",
                    "max_results": 2
                },
                "semrush": {
                    "display_limit": 3,
                    "database": "test-db"
                },
                "blog": {
                    "language": "english",
                    "word_count": {
                        "min": 100,
                        "max": 200
                    },
                    "html_format": False
                },
                "topic": "Test Topic"
            }
            json.dump(config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Load the config from the temporary file
            config = load_config(temp_file_path)
            
            # Verify the config was loaded correctly
            assert isinstance(config, Config)
            assert config.llm.model == "test-model"
            assert config.llm.temperature == 0.5
            assert config.tavily.enabled is True
            assert config.tavily.search_depth == "basic"
            assert config.tavily.max_results == 2
            assert config.semrush.display_limit == 3
            assert config.semrush.database == "test-db"
            assert config.blog.language == "english"
            assert config.blog.word_count.min == 100
            assert config.blog.word_count.max == 200
            assert config.blog.html_format is False
            assert config.topic == "Test Topic"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def test_load_config_file_not_found(self):
        # Test with a non-existent file
        with pytest.raises(RuntimeError) as excinfo:
            load_config("non_existent_file.json")
        assert "Failed to load config" in str(excinfo.value)
    
    def test_load_config_invalid_json(self):
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write("This is not valid JSON")
            temp_file_path = temp_file.name
        
        try:
            # Attempt to load the config from the invalid file
            with pytest.raises(RuntimeError) as excinfo:
                load_config(temp_file_path)
            assert "Failed to load config" in str(excinfo.value)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def test_config_default_values(self):
        # Test that default values are set correctly
        llm_config = LLMConfig()
        assert llm_config.model == "gpt-4o-mini"
        assert llm_config.temperature == 0.7
        
        tavily_config = TavilyConfig()
        assert tavily_config.enabled is True
        assert tavily_config.search_depth == "moderate"
        assert tavily_config.max_results == 3
        
        semrush_config = SemrushConfig()
        assert semrush_config.display_limit == 5
        assert semrush_config.database == "us"
        
        word_count_config = WordCountConfig()
        assert word_count_config.min == 300
        assert word_count_config.max == 500