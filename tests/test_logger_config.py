import os
import logging
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logger_config import get_logger, ColorFormatter, log_startup_context

class TestLoggerConfig:
    def test_get_logger(self):
        # We'll just test that the function returns a logger with the correct name
        # and that calling it twice with the same name returns the same logger
        unique_name = f"test_logger_{id(self)}"
        logger1 = get_logger(unique_name)
        logger2 = get_logger(unique_name)
        
        assert isinstance(logger1, logging.Logger)
        assert logger1.name == unique_name
        assert logger1 is logger2  # Same object should be returned
    
    def test_color_formatter(self):
        # Test that ColorFormatter formats messages with colors
        formatter = ColorFormatter('%(levelname)s - %(message)s')
        
        # Create a test record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Check that the message contains color codes
        assert "\033[" in formatted
        assert "INFO - Test message" in formatted
        assert formatted.endswith(formatter.RESET)
    
    @patch('src.logger_config.logging.Logger')
    def test_log_startup_context(self, mock_logger):
        # Create a mock config
        mock_config = MagicMock()
        mock_config.llm.model = "test-model"
        mock_config.llm.temperature = 0.5
        mock_config.tavily.enabled = True
        mock_config.tavily.search_depth = "test-depth"
        mock_config.tavily.max_results = 3
        mock_config.semrush.display_limit = 5
        mock_config.semrush.database = "test-db"
        mock_config.blog.language = "test-lang"
        mock_config.blog.word_count.min = 100
        mock_config.blog.word_count.max = 200
        mock_config.blog.html_format = True
        
        # Call log_startup_context
        log_startup_context(mock_logger, "Test Topic", "test-lang", mock_config)
        
        # Check that logger methods were called
        assert mock_logger.warning.called
        assert mock_logger.info.called
        
        # Check that all config values were logged
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Test Topic" in str(call) for call in info_calls)
        assert any("test-lang" in str(call) for call in info_calls)
        assert any("test-model" in str(call) for call in info_calls)
        assert any("0.5" in str(call) for call in info_calls)
        assert any("True" in str(call) for call in info_calls)
        assert any("test-depth" in str(call) for call in info_calls)
        assert any("3" in str(call) for call in info_calls)
        assert any("5" in str(call) for call in info_calls)
        assert any("test-db" in str(call) for call in info_calls)
        assert any("100" in str(call) for call in info_calls)
        assert any("200" in str(call) for call in info_calls)
    
    @patch('src.logger_config.get_logger')
    def test_log_startup_context_error_handling(self, mock_get_logger):
        # Test that log_startup_context handles errors gracefully
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Create a config object that will raise AttributeError when tavily is accessed
        class TestConfig:
            def __init__(self):
                self.llm = MagicMock()
                self.llm.model = "test-model"
                self.llm.temperature = 0.5
                # tavily is not defined, so accessing it will raise AttributeError
        
        mock_config = TestConfig()
        
        # Call log_startup_context
        log_startup_context(mock_logger, "Test Topic", "test-lang", mock_config)
        
        # Verify that the warning method was called at least once
        assert mock_logger.warning.called
        
        # The last warning call should be about the configuration fields
        # that could not be logged
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) >= 1  # At least one warning call