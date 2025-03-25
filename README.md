# SEO Blog Generator Agent

The SEO Blog Generator Agent is a Python-based tool designed to automate the creation of SEO-optimized blog posts. This project leverages state-of-the-art language models (e.g., GPT-4o-mini) and advanced research APIs (Tavily API) to produce high-quality content through a modular, graph-based workflow.

## Features

- **Modular Workflow:**  
  - **Keyword Research:** Generates targeted SEO keywords for the blog topic.
  - **Research Integration:** Retrieves and structures up-to-date research data using the Tavily API.
  - **Outline Creation:** Develops a detailed blog outline with headings and subheadings.
  - **Section Writing:** Produces engaging, HTML-formatted content for each blog section.
  - **Finalization:** Combines sections, optimizes HTML, and adds SEO metadata.
  - **Blog Finalizer Node:** Applies an additional polish to refine the overall blog post for enhanced readability and SEO performance.

- **SEO Optimization:**  
  - Tailored prompts that naturally integrate keywords.
  - Exports fully formatted HTML complete with meta tags and social sharing buttons.

- **Research Enhancement:**  
  - Uses the Tavily API for comprehensive research.
  - Provides a fallback research method for reliability.

- **Multilingual Support:**  
  - Supports content generation in both English and French.

## Architecture Overview

The project is structured into several key nodes:

1. **Keyword Research Node:**  
   Uses the OpenAI API to generate relevant SEO keywords.

2. **Research Node:**  
   Collects research data via the Tavily API and extracts structured insights.

3. **Outline Node:**  
   Creates a detailed blog outline incorporating main sections and subsections.

4. **Section Writing Node:**  
   Generates individual blog sections with HTML formatting and integrated research.

5. **Finalization Node:**  
   Combines all sections into a coherent blog post with SEO-optimized metadata.

6. **Blog Finalizer Node:**  
   Applies an extra refinement step to enhance the final HTML output.

## Installation

### Prerequisites

- Python 3.8 or higher.
- API keys for:
  - **OpenAI API** (`OPENAI_API_KEY`)
  - **Tavily API** (`API_KEY_TVLY`)

### Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/seo-blog-generator-agent.git
   cd seo-blog-generator-agent
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**  
   Create a `.env` file in the project root and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   API_KEY_TVLY=your_tavily_api_key
   ```

## Usage

To generate an SEO-optimized blog post, run the main script:
```bash
python main.py
```
This will execute the complete workflow—starting with keyword research and culminating with a fully refined HTML blog export. The final output file is saved in the designated output directory.

## Contribution

Contributions are welcome! Feel free to fork this repository, make improvements, and submit pull requests. For significant changes, please open an issue first to discuss your proposed modifications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
