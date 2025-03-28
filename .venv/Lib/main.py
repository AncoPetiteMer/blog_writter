import os
from datetime import datetime
from typing import Dict, List, Any, TypedDict, Callable
import json
import re
import requests
import logging
from openai import OpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from dotenv import load_dotenv
from research_tools import search_with_tavily, search_with_wikipedia, search_with_exa



# Set your OpenAI API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('API_KEY_TVLY')
os.environ["EXA_API_KEY"] = os.getenv('EXA_API_KEY')
os.environ["SEMRUSH_API_KEY"] = os.getenv('SEMRUSH_API_KEY')

# Logger
from logger_config import get_logger
logger = get_logger("seo_blog")

# Define our state
class BlogState(TypedDict):
    topic: str
    keywords: List[str]
    research: Dict[str, Any]
    outline: List[Dict[str, Any]]
    current_section: int
    sections_content: Dict[int, str]
    final_blog: str
    metadata: Dict[str, Any]
    debug_info: Dict[str, Any]  # Added for debugging

# Initialize OpenAI client
client = OpenAI()

def call_llm(prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
    """Helper function to call the OpenAI API directly."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature
    )

    return response.choices[0].message.content

def extract_json(content: str) -> Any:
    """Extract JSON from a string response."""
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

def keyword_research(state: BlogState) -> BlogState:
    """Generate SEO keywords based on the topic using LLM and Semrush API."""
    import csv
    import io

    topic = state.get("topic", "")
    logger.info(f"Starting keyword research for topic: '{topic}'")

    if not topic:
        return {
            "keywords": ["no-topic-provided"],
            "debug_info": {"stage": "keyword_research", "status": "error"}
        }

    semrush_keywords = []
    try:
        import requests
        semrush_api_key = os.getenv("SEMRUSH_API_KEY")
        params = {
            "type": "phrase_related",
            "key": semrush_api_key,
            "phrase": topic,
            "database": "us",
            "export_columns": "Ph",
            "display_limit": 5,
        }
        response = requests.get("https://api.semrush.com/", params=params)

        if response.ok:
            reader = csv.reader(io.StringIO(response.text), delimiter=';')
            next(reader, None)  # skip header row
            semrush_keywords = [row[0] for row in reader if row and row[0]]
            logger.info(f"SEMRush keywords: {semrush_keywords}")
        else:
            logger.warning(f"SEMRush API returned non-OK: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"SEMRush API error: {e}", exc_info=True)

    try:
        system_prompt = f"You are an SEO expert. Generate keywords for: {topic}"
        prompt = f"""
        Generate 2-3 relevant SEO keywords for: "{topic}"
        Return ONLY a JSON array of strings.
        Example: ["keyword 1", "keyword 2", "keyword 3"]
        """
        llm_response = call_llm(prompt, system_prompt)
        llm_keywords = extract_json(llm_response)
        combined_keywords = list(set(llm_keywords + semrush_keywords))[:5]
        logger.info(f"Combined keywords: {combined_keywords}")
        return {
            "keywords": combined_keywords,
            "debug_info": {
                "stage": "keyword_research",
                "status": "success",
                "keywords": combined_keywords
            }
        }
    except Exception as e:
        logger.error(f"LLM error: {str(e)}", exc_info=True)
        fallback_keywords = semrush_keywords or [f"{topic} best practices", f"{topic} guide", f"{topic} tips"]
        return {
            "keywords": fallback_keywords,
            "debug_info": {
                "stage": "keyword_research",
                "status": "fallback",
                "keywords": fallback_keywords
            }
        }




def create_outline(state: BlogState) -> BlogState:
    """Create a detailed blog post outline based on research."""
    logger.info("Creating blog outline")

    keywords_str = ", ".join(state["keywords"])
    research_str = json.dumps(state["research"], indent=2)

    prompt = f"""
You are a content strategist. Create a detailed blog post outline optimized for SEO.

Create a detailed outline for the following topic:

Topic: {state['topic']}
Keywords: {keywords_str}
Research: {research_str}

Your outline should:
1. Include a compelling headline
2. Have an introduction that hooks the reader
3. Contain 2-3 main sections with subheadings
4. Include a conclusion
5. All sections should naturally incorporate the target keywords
6. Suggest places to include statistics or examples from the research

Return a JSON array of objects with this format:
[
  {{"title": "Introduction", "subsections": []}},
  {{"title": "Section 1", "subsections": ["Subsection 1A", "Subsection 1B"]}},
  {{"title": "Conclusion", "subsections": []}}
]
    """

    response = call_llm(prompt)
    outline = extract_json(response)

    # Print the outline for testing
    logger.info("Blog outline created:")
    for i, section in enumerate(outline):
        subsections = ", ".join(section.get("subsections", []))
        logger.info(f"  Section {i+1}: {section['title']} - Subsections: {subsections if subsections else 'None'}")

    return {
        "outline": outline,
        "current_section": 0,
        "sections_content": {},
        "debug_info": {"stage": "create_outline", "status": "success", "outline_summary": [section["title"] for section in outline]}
    }


def write_section(state: BlogState) -> BlogState:
    """Write a specific section of the blog post with Tavily research and HTML formatting."""
    current_section = state["current_section"]
    logger.info(f"Writing section {current_section + 1} of {len(state['outline'])}")

    section = state["outline"][current_section]

    # Get previous content for context
    previous_context = ""
    for i in range(current_section):
        if i in state["sections_content"]:
            previous_context += f"Section {i+1}: {state['sections_content'][i]}\n\n"

    # Perform Tavily search for this specific section
    from tavily import TavilyClient

    tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    search_query = f"{state['topic']} {section['title']} {', '.join(state['keywords'])}"

    tavily_results = tavily_client.search(
        query=search_query,
        search_depth="advanced",
        max_results=3
    )
    logger.info(f"  Tavily search results for writtings @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ {tavily_results} @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    # Extract sources and content from Tavily results
    sources = [{"title": result.get("title", ""), "url": result.get("url", "")}
               for result in tavily_results.get("results", [])]
    tavily_content = "\n".join([result.get("content", "") for result in tavily_results.get("results", [])])

    # Prepare prompt with Tavily research and HTML instructions
    keywords_str = ", ".join(state["keywords"])
    subsections_str = ", ".join(section.get("subsections", []))

    prompt = f"""
    You are an SEO-savvy content writer. Write engaging, informative content for blog sections with HTML formatting.

    Blog Topic: {state['topic']}
    Target Keywords: {keywords_str}

    Previous Content: {previous_context}
    Current Section: {section['title']}
    Subsections: {subsections_str}

    Research: {tavily_content}

    Guidelines:
    - Write engaging content in French (300-500 words)
    - Format the content with proper HTML structure:
      - Use <h2> for the main section heading
      - Use <h3> for subsection headings
      - Use <p> for paragraphs
      - Use <ul> and <li> for lists
      - Use <a href="URL">text</a> for links
      - Use <strong> for emphasis on important points
    - Naturally incorporate these keywords: {keywords_str}
    - Use a conversational yet authoritative tone
    - Include relevant information from the research
    - Add a smooth transition to the next section if applicable
    - Do not include any introduction like "Here's the HTML content" - just provide the formatted HTML
    """

    # Generate content with HTML formatting
    section_content = call_llm(prompt)

    # Add sources at the end with HTML formatting
    if sources:
        sources_html = "<div class='sources'>\n<h3>Sources</h3>\n<ol>\n"
        for source in sources:
            sources_html += f"<li><a href='{source['url']}' target='_blank'>{source['title']}</a></li>\n"
        sources_html += "</ol>\n</div>"
        section_content += "\n\n" + sources_html

    # Log preview
    logger.info(f"Section {current_section + 1} ({section['title']}) content preview:")
    preview = section_content[:150] + "..." if len(section_content) > 150 else section_content
    logger.info(f"  {preview.replace(chr(10), ' ').replace(chr(13), '')}")

    # Update state
    sections_content = state["sections_content"].copy()
    sections_content[current_section] = section_content

    return {
        "sections_content": sections_content,
        "current_section": current_section + 1,
        "debug_info": {
            "stage": "write_section",
            "status": "success",
            "section": current_section,
            "section_title": section['title'],
            "content_length": len(section_content),
            "sources_count": len(sources)
        }
    }

def should_continue_writing(state: BlogState) -> str:
    """Decide if we should continue writing sections or finalize the blog."""
    current = state["current_section"]
    total = len(state["outline"])

    logger.info(f"Section completion check: {current}/{total} sections completed")

    if current < total:
        return "continue_writing"
    else:
        return "finalize"

def finalize_blog(state: BlogState) -> BlogState:
    """Combine all sections and optimize the final blog post."""
    logger.info("Finalizing blog post")

    # Save all the section contents for debugging
    logger.info("Section content details:")
    for i, section in enumerate(state["outline"]):
        if i in state["sections_content"]:
            content_len = len(state["sections_content"][i])
            logger.info(f"  Section {i+1}: {section['title']} - {content_len} characters")
        else:
            logger.info(f"  Section {i+1}: {section['title']} - MISSING CONTENT")

    sections_content = state["sections_content"]
    all_content = ""

    for i in range(len(state["outline"])):
        if i in sections_content:
            all_content += sections_content[i] + "\n\n"
        else:
            logger.warning(f"Missing content for section {i+1}")

    keywords_str = ", ".join(state["keywords"])
    logger.info(f"Le contenu finale est: ###############################{all_content} ########################################")
    final_prompt = f"""
    You are an SEO editor and HTML formatter. Optimize this blog post for readability, SEO, and proper HTML structure.

    Review and optimize this blog post:

    Topic: {state['topic']}
    Keywords: {keywords_str}

    Content:
    {all_content}

    Your tasks:
    1. Format the entire blog post with proper HTML, including:
      - <h1> for the main title
      - <h2> for section headings
      - <h3> for subsection headings
      - <p> for paragraphs
      - <ul>/<ol> and <li> for lists
      - <blockquote> for quotes
      - <strong> for emphasis on important points
      - <a href="URL">text</a> for links
      - Add proper <meta> tags in the <head> section for SEO

    2. Ensure the HTML structure is clean and semantic with:
      - Proper indentation
      - Appropriate <div> containers for sections
      - <header>, <main>, <footer> structural elements
 
    3. Content optimization:
      - Add a compelling introduction and conclusion if they're not strong enough
      - Ensure smooth transitions between sections
      - Naturally incorporate keywords without stuffing
      - Add a table of contents at the beginning with anchor links

    4. Add these finishing touches:
      - Include an author bio section at the end
      - Add social sharing buttons HTML
      - Include "Related Posts" section HTML
      - Add a call-to-action for newsletter signup or comments

    5. Return ONLY the complete HTML document with all formatting included, no explanations.
      Begin with <!DOCTYPE html> and include all necessary HTML structure.
    """

    final_blog = call_llm(final_prompt)

    logger.info(f"Final blog post created with {len(final_blog)} characters")

    meta_prompt = f"""
You are an SEO specialist. Create metadata for a blog post.

Create metadata for the following blog post:

Topic: {state['topic']}
Keywords: {keywords_str}

Create these metadata elements:
1. SEO Title Tag (60-70 characters)
2. Meta Description (150-160 characters)
3. H1 Headline (compelling headline for the page)

Return a JSON object with these fields:
{{
    "seo_title": "Your SEO title here",
    "meta_description": "Your meta description here",
    "h1_headline": "Your H1 headline here"
}}
    """

    metadata_response = call_llm(meta_prompt)
    metadata = extract_json(metadata_response)

    logger.info(f"Metadata generated: Title: '{metadata.get('seo_title', '')}'")

    return {
        "final_blog": final_blog,
        "metadata": metadata,
        "debug_info": {
            "stage": "finalize_blog",
            "status": "success",
            "blog_length": len(final_blog),
            "metadata": metadata
        }
    }

def build_seo_blog_writer() -> StateGraph:
    workflow = StateGraph(BlogState)

    # Add nodes
    workflow.add_node("keyword_research", keyword_research)
    workflow.add_node("research_topic", research_topic)
    workflow.add_node("create_outline", create_outline)
    workflow.add_node("write_section", write_section)
    workflow.add_node("finalize_blog", finalize_blog)

    # Add edges
    workflow.add_edge("keyword_research", "research_topic")
    workflow.add_edge("research_topic", "create_outline")
    workflow.add_edge("create_outline", "write_section")
    workflow.add_conditional_edges(
        "write_section",
        should_continue_writing,
        {
            "continue_writing": "write_section",
            "finalize": "finalize_blog"
        }
    )
    workflow.add_edge("finalize_blog", END)

    # Set entry point
    workflow.set_entry_point("keyword_research")

    return workflow

# Update the generate_seo_blog function to correctly handle the final state

def generate_seo_blog(topic: str, debug: bool = False) -> Dict[str, Any]:
    """Generate a complete SEO-optimized blog post for the given topic."""
    logger.info(f"Generating SEO blog for topic: '{topic}'")

    initial_state = BlogState(
        topic=topic,
        keywords=[],
        research={},
        outline=[],
        current_section=0,
        sections_content={},
        final_blog="",
        metadata={},
        debug_info={}
    )

    graph = build_seo_blog_writer().compile()
    display(Image(graph.get_graph().draw_mermaid_png()))

    # Execute the graph and get the final state directly
    config = {"configurable": {"thread_id": "blog_gen_" + str(hash(topic))}}
    final_state = graph.invoke(initial_state, config=config)

    # Print events if debug is enabled
    if debug:
        logger.info(f"Graph execution completed")

    # Create result from final state
    result = {
        "blog_content": final_state.get("final_blog", ""),
        "metadata": final_state.get("metadata", {}),
        "keywords": final_state.get("keywords", []),
        "outline": final_state.get("outline", []),
        "sections": final_state.get("sections_content", {}),
        "debug_info": final_state.get("debug_info", {})
    }

    return result

def print_blog_sections(result: Dict[str, Any]) -> None:
    """Pretty-print the blog sections for testing."""
    print("\n" + "="*80)
    print(f"BLOG SECTIONS FOR: {result.get('metadata', {}).get('h1_headline', 'Untitled Blog')}")
    print("="*80)

    sections = result.get("sections", {})
    outline = result.get("outline", [])

    if not sections:
        print("No sections content available.")
        return

    print(f"Total sections: {len(outline)}")

    for i, section in enumerate(outline):
        if i in sections:
            title = section.get("title", f"Section {i+1}")
            print(f"\n{'#'*40} SECTION {i+1}: {title} {'#'*40}\n")
            print(sections[i])
            print("\n" + "-"*80)
        else:
            print(f"\nSECTION {i+1}: {section.get('title', 'Untitled')} - CONTENT MISSING")

    print("\n" + "="*80)
    print("METADATA:")
    print(f"SEO Title: {result.get('metadata', {}).get('seo_title', 'No title')}")
    print(f"Meta Description: {result.get('metadata', {}).get('meta_description', 'No description')}")
    print(f"H1 Headline: {result.get('metadata', {}).get('h1_headline', 'No headline')}")
    print("="*80)

def export_to_html(result: Dict[str, Any], output_path: str = None) -> str:
    """
    Export the blog post to an HTML file with proper formatting and styling.

    Args:
        result: The blog generation result dictionary
        output_path: Path where to save the HTML file (if None, uses the blog title)

    Returns:
        Path to the saved HTML file
    """
    blog_content = result.get("blog_content", "")
    metadata = result.get("metadata", {})
    keywords = result.get("keywords", [])

    title = metadata.get("h1_headline", "blog-post")
    sanitized_title = "".join(c if c.isalnum() else "-" for c in title).lower()

    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = f"{sanitized_title}-{timestamp}.html"

    seo_title = metadata.get("seo_title", title)
    meta_description = metadata.get("meta_description", "")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{seo_title}</title>
    <meta name="description" content="{meta_description}" />
</head>
<body>
    <div class="metadata">
        <p><strong>SEO Title :</strong> {seo_title}</p>
        <p><strong>Meta Description :</strong> {meta_description}</p>
        <p><em>Mots-clés :</em> {', '.join(keywords)}</p>
    </div>

    <h1>{title}</h1>

    {blog_content}

    <footer>
        <p><small>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </footer>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Blog exported as HTML to: {output_path}")
    return output_path


def research_topic_with_tavily(state: BlogState) -> BlogState:
    """Research the topic using Tavily API and gather relevant information."""
    import os
    from tavily import TavilyClient

    topic = state["topic"]
    keywords = state["keywords"]

    logger.info(f"Researching topic with Tavily: '{topic}'")

    try:
        # Initialize Tavily client directly
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("Tavily API key is required. Set the TAVILY_API_KEY environment variable.")

        client = TavilyClient(api_key)

        # Create search queries based on topic and keywords
        search_queries = [topic]
        if keywords:
            # Add topic + each keyword as separate queries
            for keyword in keywords[:3]:  # Limit to avoid too many API calls
                search_queries.append(f"{topic} {keyword}")

        # Store all search results
        all_results = []

        for query in search_queries:
            try:
                logger.info(f"Performing Tavily search: {query}")
                # search_results = client.search(
                #     query=query,
                #     search_depth="moderate",
                #     max_results=5
                # )

                search_results = client.search(query)
                logger.info(f"  Tavily search results for topics ######################### {search_results} ################################")
                if "results" in search_results:
                    all_results.extend(search_results["results"])
            except Exception as e:
                logger.error(f"Tavily search error for query '{query}': {str(e)}")

        # Extract and structure the information from search results
        structured_data = {
            "key_facts": [],
            "statistics": [],
            "trends": [],
            "audience_insights": [],
            "competitors": [],
            "sources": []
        }

        # Add unique URLs to sources
        unique_urls = set()
        for result in all_results:
            url = result.get("url")
            if url and url not in unique_urls:
                unique_urls.add(url)
                structured_data["sources"].append({
                    "title": result.get("title", "Untitled"),
                    "url": url
                })

        # Create a comprehensive query for the LLM to extract structured information
        combined_content = "\n\n".join([
            f"Title: {result.get('title', 'Untitled')}\n"
            f"Content: {result.get('content', 'No content')}"
            for result in all_results[:10]  # Limit to avoid token limits
        ])

        # Use the LLM to extract structured information
        if combined_content:
            extraction_prompt = f"""
Analyze the following research on "{topic}" and extract structured information.

RESEARCH CONTENT:
{combined_content}

Extract and categorize the information into these categories:
1. Key Facts: Important facts about {topic}
2. Statistics: Numerical data and statistics related to {topic}
3. Trends: Current and emerging trends in {topic}
4. Audience Insights: Information about the target audience, demographics, or user behaviors
5. Competitors: Major competitors or competing concepts in this space

Return the information in this JSON format:
{{
    "key_facts": ["fact 1", "fact 2", ...],
    "statistics": ["statistic 1", "statistic 2", ...],
    "trends": ["trend 1", "trend 2", ...],
    "audience_insights": ["insight 1", "insight 2", ...],
    "competitors": ["competitor 1", "competitor 2", ...]
}}

Only include information actually mentioned in the research content. Use direct quotes where possible.
"""
            extraction_response = call_llm(extraction_prompt)
            extracted_data = extract_json(extraction_response)

            # Update structured data with extracted information
            for key in structured_data:
                if key in extracted_data and key != "sources":
                    structured_data[key] = extracted_data[key]

        # Check if we got enough results
        total_items = sum(len(v) for k, v in structured_data.items()
                          if k != "sources" and isinstance(v, list))

        if total_items < 10:
            logger.warning(f"Limited Tavily results ({total_items} items). Using fallback research method.")
            return research_topic(state)

        logger.info(f"Tavily research completed with {total_items} data points and "
                   f"{len(structured_data.get('sources', []))} sources")

        return {
            **state,
            "research": structured_data,
            "debug_info": {
                "stage": "research_topic_with_tavily",
                "status": "success",
                "research_summary": {
                    k: len(v) for k, v in structured_data.items()
                    if isinstance(v, list)
                },
                "sources_count": len(structured_data.get("sources", []))
            }
        }

    except Exception as e:
        logger.error(f"Tavily research error: {str(e)}")
        logger.info("Falling back to regular research method")
        # Fall back to the regular research method
        return research_topic(state)


def research_topic(state: BlogState) -> BlogState:
    """Research the topic and gather relevant information."""
    logger.info(f"Researching topic: '{state['topic']}'")

    prompt = f"""
You are a research specialist. Gather important information about the given topic.

Topic: {state['topic']}
Relevant Keywords: {', '.join(state['keywords'])}

Provide the following information in JSON format:
{{
    "key_facts": ["fact 1", "fact 2", ...],
    "statistics": ["statistic 1", "statistic 2", ...],
    "trends": ["trend 1", "trend 2", ...],
    "audience_insights": ["insight 1", "insight 2", ...],
    "competitors": ["competitor 1", "competitor 2", ...]
}}
    """

    response = call_llm(prompt)
    research = extract_json(response)

    logger.info(f"Research completed with {sum(len(v) for v in research.values() if isinstance(v, list))} data points")

    return {
        "research": research,
        "debug_info": {"stage": "research_topic", "status": "success", "research_summary": {k: len(v) for k, v in research.items() if isinstance(v, list)}}
    }

def build_enhanced_seo_blog_writer() -> StateGraph:
    """Build the enhanced version of the SEO blog writer workflow."""
    workflow = StateGraph(BlogState)

    # Add nodes
    workflow.add_node("keyword_research", keyword_research)
    workflow.add_node("research_topic", research_topic_with_tavily)  # Use Tavily-enhanced research
    workflow.add_node("create_outline", create_outline)
    workflow.add_node("write_section", write_section)
    workflow.add_node("finalize_blog", finalize_blog)

    # Add edges
    workflow.add_edge("keyword_research", "research_topic")
    workflow.add_edge("research_topic", "create_outline")
    workflow.add_edge("create_outline", "write_section")
    workflow.add_conditional_edges(
        "write_section",
        should_continue_writing,
        {
            "continue_writing": "write_section",
            "finalize": "finalize_blog"
        }
    )
    workflow.add_edge("finalize_blog", END)

    # Set entry point
    workflow.set_entry_point("keyword_research")

    return workflow

def generate_enhanced_seo_blog(
    topic: str,
    language: str = "english",
    use_tavily: bool = True,
    export_html: bool = True,
    output_dir: str = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Generate a complete enhanced SEO-optimized blog post for the given topic.

    Args:
        topic: The blog topic
        language: Language for the content ("english" or "french")
        use_tavily: Whether to use Tavily API for research
        export_html: Whether to export the result as HTML
        output_dir: Directory to save the HTML export (if None, uses current directory)
        debug: Whether to enable debug logging

    Returns:
        Result dictionary with blog content and metadata
    """
    logger.info(f"Generating enhanced SEO blog for topic: '{topic}' in {language}")

    # Prepend language preference to topic if not English
    topic_with_lang = topic
    if language.lower() != "english":
        topic_with_lang = f"[{language.upper()}] {topic}"

    initial_state = BlogState(
        topic=topic_with_lang,
        keywords=[],
        research={},
        outline=[],
        current_section=0,
        sections_content={},
        final_blog="",
        metadata={},
        debug_info={"language": language}
    )

    # Use the appropriate graph based on Tavily preference
    if use_tavily:
        graph = build_enhanced_seo_blog_writer().compile()
    else:
        graph = build_seo_blog_writer().compile()

    # Display the graph visualization
    # display(Image(graph.get_graph().draw_mermaid_png()))

    # Execute the graph and get the final state
    config = {
        "configurable": {
            "thread_id": f"blog_gen_{language}_{str(hash(topic))}"
        }
    }
    final_state = graph.invoke(initial_state, config=config)

    # Print events if debug is enabled
    if debug:
        logger.info(f"Graph execution completed")

    # Create result from final state
    result = {
        "blog_content": final_state.get("final_blog", ""),
        "metadata": final_state.get("metadata", {}),
        "keywords": final_state.get("keywords", []),
        "outline": final_state.get("outline", []),
        "sections": final_state.get("sections_content", {}),
        "research": final_state.get("research", {}),
        "debug_info": final_state.get("debug_info", {})
    }

    # Export to HTML if requested
    if export_html:
        html_path = None
        if output_dir:
            # Make sure the output directory exists
            import os
            os.makedirs(output_dir, exist_ok=True)
            # Create the output path
            filename = f"{language}-{'-'.join(topic.split()[:5])}.html"
            html_path = os.path.join(output_dir, filename)

        # Export the blog to HTML
        exported_path = export_to_html(result, html_path)
        result["html_export_path"] = exported_path

    return result

# Basic usage with all enhancements
result = generate_enhanced_seo_blog(
    topic="Machine Learning Applications in Healthcare",
    language="english",
    use_tavily=True,
    export_html=True,
    output_dir="blog_exports"
)

# Print the HTML export path
print(f"Blog exported to: {result['html_export_path']}")

