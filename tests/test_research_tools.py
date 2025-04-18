import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.research_tools import search_with_tavily, search_with_wikipedia, search_with_exa

class TestResearchTools:
    @patch('tavily.TavilyClient')
    def test_search_with_tavily(self, mock_tavily_client):
        # Set up the mock
        mock_client_instance = MagicMock()
        mock_tavily_client.return_value = mock_client_instance
        
        # Set up the mock search results
        mock_results = {
            "results": [
                {"content": "keyword1 keyword2 some text"},
                {"content": "more text with keyword3 and keyword4"}
            ]
        }
        mock_client_instance.search.return_value = mock_results
        
        # Call the function
        result = search_with_tavily("test topic")
        
        # Verify the results
        assert isinstance(result, list)
        assert len(result) <= 5  # Should return at most 5 keywords
        
        # Verify the TavilyClient was called correctly
        mock_tavily_client.assert_called_once()
        mock_client_instance.search.assert_called_once()
        args, kwargs = mock_client_instance.search.call_args
        assert "test topic" in kwargs.get("query", "")
        assert kwargs.get("search_depth") == "advanced"
        assert kwargs.get("max_results") == 3
    
    @patch('wikipediaapi.Wikipedia')
    def test_search_with_wikipedia(self, mock_wikipedia_class):
        # Set up the mock
        mock_wiki = MagicMock()
        mock_wikipedia_class.return_value = mock_wiki
        
        # Set up the mock page
        mock_page = MagicMock()
        mock_page.exists.return_value = True
        mock_page.summary = "This is a test summary with some keywords and phrases"
        mock_wiki.page.return_value = mock_page
        
        # Call the function
        result = search_with_wikipedia("test topic")
        
        # Verify the results
        assert isinstance(result, list)
        assert len(result) <= 5  # Should return at most 5 keywords
        
        # Verify the Wikipedia API was called correctly
        mock_wikipedia_class.assert_called_once_with("en")
        mock_wiki.page.assert_called_once_with("test topic")
    
    @patch('wikipediaapi.Wikipedia')
    def test_search_with_wikipedia_nonexistent_page(self, mock_wikipedia_class):
        # Set up the mock
        mock_wiki = MagicMock()
        mock_wikipedia_class.return_value = mock_wiki
        
        # Set up the mock page to not exist
        mock_page = MagicMock()
        mock_page.exists.return_value = False
        mock_wiki.page.return_value = mock_page
        
        # Call the function
        result = search_with_wikipedia("nonexistent topic")
        
        # Verify the results
        assert result == []  # Should return an empty list
    
    def test_search_with_exa(self):
        # This is a simple function that doesn't use external APIs
        result = search_with_exa("test topic")
        
        # Verify the results
        assert isinstance(result, list)
        assert len(result) == 3
        assert "test topic analysis" in result
        assert "test topic insights" in result
        assert "test topic examples" in result