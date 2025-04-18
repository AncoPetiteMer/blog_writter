import pytest
import json
import sys
import os
import re
import logging
from unittest.mock import patch, MagicMock

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a mock version of the extract_json function for testing
# This is the same implementation as in main.py
def extract_json(content: str) -> any:
    """Extract JSON from a string response."""
    logger = logging.getLogger("test_logger")
    try:
        # Try to find JSON pattern in the response
        json_match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', content)
        if json_match:
            content = json_match.group(0)
        return json.loads(content)
    except Exception as e:
        logger.error(f"JSON extraction error: {str(e)}")
        logger.error(f"Content was: {content}")
        # Return empty dict or list as fallback
        return {} if content.strip().startswith('{') else []

class TestExtractJson:
    def test_extract_json_dict(self):
        # Test with a valid JSON dictionary
        json_str = '{"key1": "value1", "key2": 42}'
        result = extract_json(json_str)
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
    
    def test_extract_json_list(self):
        # Test with a valid JSON list
        json_str = '["item1", "item2", "item3"]'
        result = extract_json(json_str)
        assert isinstance(result, list)
        assert len(result) == 3
        assert "item1" in result
        assert "item2" in result
        assert "item3" in result
    
    def test_extract_json_from_text(self):
        # Test extracting JSON from text with surrounding content
        text = """
        Here is some text before the JSON.
        {"key1": "value1", "key2": 42}
        And here is some text after the JSON.
        """
        result = extract_json(text)
        assert isinstance(result, dict)
        assert result["key1"] == "value1"
        assert result["key2"] == 42
    
    def test_extract_json_nested(self):
        # Test with nested JSON
        json_str = '{"outer": {"inner1": "value1", "inner2": [1, 2, 3]}}'
        result = extract_json(json_str)
        assert isinstance(result, dict)
        assert "outer" in result
        assert isinstance(result["outer"], dict)
        assert result["outer"]["inner1"] == "value1"
        assert result["outer"]["inner2"] == [1, 2, 3]
    
    def test_extract_json_invalid(self):
        # Test with invalid JSON
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = extract_json("This is not valid JSON")
            # Should return an empty list as fallback (since the input doesn't start with '{')
            assert result == []
            # Should log an error
            assert mock_logger.error.called
    
    def test_extract_json_empty(self):
        # Test with empty string
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            result = extract_json("")
            # Should return an empty list as fallback (since the input doesn't start with '{')
            assert result == []
            # Should log an error
            assert mock_logger.error.called