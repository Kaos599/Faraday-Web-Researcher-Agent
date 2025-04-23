"""
This module contains wrapper functions for various search and information gathering tools
used by the Web Research Agent system, refactored as Langchain Tools.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Type
import requests
from . import config  # Import from the current package

# Import schemas and prompts
# Remove FactCheckResult, EvidenceSource if not used directly here
from .schemas import ResearchReport, Source, IntermediateStep, ResearchRequest, ErrorResponse, GeminiParsedOutput
from .prompts import QUERY_DECOMPOSITION_TEMPLATE # Import the new template

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool, BaseTool
from .config import get_primary_llm, get_azure_openai_parser_llm
from langchain.tools import Tool # Keep Tool for wrapping functions if needed
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field # Use v1 for tool compatibility
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.document_loaders import WebBaseLoader # For Web Scraping
from newsapi import NewsApiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Internal Helper Functions (like parsing) ---
# Update context from claim to query
def parse_gemini_output_with_llm(gemini_raw_output: str, query: str) -> Optional[GeminiParsedOutput]:
    """
    Uses the secondary Azure OpenAI LLM to parse the raw text output from the
    Gemini+GoogleSearch call into a structured format. Internal helper function.

    Args:
        gemini_raw_output: The raw string output from perform_gemini_google_search.
        query: The original research query, provided for context.

    Returns:
        A GeminiParsedOutput Pydantic object, or None if parsing fails or LLM is unavailable.
    """
    parser_llm = get_azure_openai_parser_llm() # Get client inside function
    if not parser_llm:
        logger.error("Azure OpenAI Parser LLM is not available for Gemini parsing.")
        return None

    parser = JsonOutputParser(pydantic_object=GeminiParsedOutput)

    # Update prompt template to use 'query' instead of 'claim'
    prompt_template = ChatPromptTemplate.from_template(
        """You are an expert assistant tasked with parsing the output of a Google Search-enabled Gemini model call.
        The Gemini model was asked to investigate the following research query: '{query}'
        Its raw output, potentially containing summaries, facts, and source information, is provided below.
        Your goal is to extract the key information and structure it into a JSON object matching the requested format.

        Focus on identifying:
        1.  A concise summary of the findings regarding the query.
        2.  A list of key facts or pieces of information presented.
        3.  A list of URLs identified as sources in the text.

        Raw Gemini Output:
        ---
        {gemini_raw_output}
        ---

        Format Instructions:
        {format_instructions}
        """
    )

    chain = prompt_template | parser_llm | parser
    logger.info("Attempting to parse Gemini output using Azure OpenAI Parser LLM.")

    try:
        parsed_result_dict = chain.invoke({ # Store as dict first
            "query": query, # Pass query here
            "gemini_raw_output": gemini_raw_output,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully parsed Gemini output. Filtering sources...")

        # Filter sources within the parsed dictionary
        filtered_sources = []
        if "sources" in parsed_result_dict and isinstance(parsed_result_dict["sources"], list):
            for source_url in parsed_result_dict["sources"]:
                # Basic URL validation (check if string and starts with http/https)
                if isinstance(source_url, str) and source_url.strip().lower().startswith(("http://", "https://")):
                    filtered_sources.append(source_url.strip())
                else:
                    logger.warning(f"Filtering out invalid source URL from Gemini output: {source_url}")
            parsed_result_dict["sources"] = filtered_sources # Update dict with filtered list
        else:
             logger.warning("No 'sources' list found in parsed Gemini output or it's not a list.")
             parsed_result_dict["sources"] = [] # Ensure sources key exists as empty list

        # Now create the Pydantic object from the filtered dictionary
        return GeminiParsedOutput(**parsed_result_dict)
    except OutputParserException as ope:
         logger.error(f"Failed to parse Gemini output. Parser Error: {ope}. Raw output was:\\n{gemini_raw_output[:500]}...")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Gemini output parsing: {e}")
        return None

# --- Tool Input Schemas ---

class SearchInput(BaseModel):
    query: str = Field(description="The search query string.")

class WikidataInput(BaseModel):
    query: str = Field(description="A specific entity name (e.g., 'Eiffel Tower') or concept to query Wikidata for structured data.")

class FirecrawlInput(BaseModel):
    url: str = Field(description="A single, specific URL to scrape for its main content in Markdown format.")

class NewsSearchInput(BaseModel):
    query: str = Field(description="Keywords or phrase to search for in recent news articles.")
    language: str = Field(default='en', description="The 2-letter ISO 639-1 code of the language.")
    page_size: int = Field(default=10, description="Number of news results (max 100).")


# --- Query Decomposition Tool (Refactored) ---
class QueryDecompositionInput(BaseModel): # Renamed
    query: str = Field(..., description="The complex research query to decompose into sub-topics or questions.") # Renamed field

class QueryDecompositionOutput(BaseModel): # Renamed
    sub_queries: List[str] = Field(..., description="A list of simpler, searchable sub-queries or sub-topics derived from the original query.") # Renamed field

# Remove internal DECOMPOSITION_PROMPT definition, use imported QUERY_DECOMPOSITION_TEMPLATE

class QueryDecompositionTool(BaseTool): # Renamed
    """Tool to decompose complex research queries."""
    name: str = "query_decomposition_tool" # Renamed
    description: str = (
        "Decomposes a complex research query (containing multiple facets, entities, or questions) into a list of simpler, "
        "focused sub-queries or sub-topics. Use this *first* on complex queries to guide the research strategy and generate targeted searches for other tools."
    ) # Updated description
    args_schema: Type[BaseModel] = QueryDecompositionInput # Updated schema

    def _run(self, query: str) -> Dict[str, Any]: # Use query as arg name
        """Executes the query decomposition."""
        logger.info(f"Executing Query Decomposition Tool for query: '{query}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for Query Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        # Use a ListOutputParser or similar if the prompt reliably returns a list string
        # For now, assume JSON output parser based on prompt examples
        # If using QUERY_DECOMPOSITION_TEMPLATE which outputs a list string:
        # from langchain.output_parsers import CommaSeparatedListOutputParser
        # parser = CommaSeparatedListOutputParser()
        # OR use StrOutputParser and parse the string list manually
        from langchain_core.output_parsers import StrOutputParser
        import ast
        prompt = QUERY_DECOMPOSITION_TEMPLATE # Use imported prompt
        chain = prompt | llm | StrOutputParser()

        try:
            result_str = chain.invoke({"query": query})
            # Attempt to parse the string representation of a list
            try:
                sub_queries_list = ast.literal_eval(result_str.strip())
                if not isinstance(sub_queries_list, list):
                    raise ValueError("Parsed result is not a list")
                result = {"sub_queries": sub_queries_list}
                logger.info(f"Query decomposition successful. Found {len(result.get('sub_queries', []))} sub-queries.")
            except (SyntaxError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse sub-query list string: {parse_err}. Raw result: {result_str}")
                # Fallback: return the raw string wrapped in an error or a specific format
                # return {"error": f"Failed to parse sub-query list: {result_str}"}
                # Or try to return it as a single-item list if appropriate
                result = {"sub_queries": [result_str]} # Or handle error differently
            return result
        except Exception as e:
            logger.error(f"Error during query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Query decomposition failed: {str(e)}"}

    async def _arun(self, query: str) -> Dict[str, Any]: # Use query as arg name
        """Async execution."""
        logger.info(f"Executing async Query Decomposition Tool for query: '{query}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for async Query Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        from langchain_core.output_parsers import StrOutputParser
        import ast
        prompt = QUERY_DECOMPOSITION_TEMPLATE
        chain = prompt | llm | StrOutputParser()

        try:
            result_str = await chain.ainvoke({"query": query})
            try:
                sub_queries_list = ast.literal_eval(result_str.strip())
                if not isinstance(sub_queries_list, list):
                    raise ValueError("Parsed result is not a list")
                result = {"sub_queries": sub_queries_list}
                logger.info(f"Async query decomposition successful. Found {len(result.get('sub_queries', []))} sub-queries.")
            except (SyntaxError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse async sub-query list string: {parse_err}. Raw result: {result_str}")
                result = {"sub_queries": [result_str]} # Fallback
            return result
        except Exception as e:
            logger.error(f"Error during async query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Async query decomposition failed: {str(e)}"}

# --- Search & Retrieval Tools --- (Descriptions slightly adjusted)

@tool(args_schema=SearchInput)
def tavily_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Performs a comprehensive web search using Tavily. Excellent for broad exploration or finding diverse sources on a research query or sub-topic.
    Returns a list of relevant documents with URLs, titles, and content snippets.
    Good starting point for most research tasks.
    """
    tavily_client = config.get_tavily_client()
    if not tavily_client:
        logger.error("Tavily client is not available.")
        return {"error": "Tavily client unavailable."}
    try:
        logger.info(f"Performing Tavily search for query: '{query}'")
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        if isinstance(response, list):
            response_dict = {"results": response}
        else:
            response_dict = response
        logger.info(f"Tavily search completed. Found {len(response_dict.get('results', []))} results.")
        if 'results' in response_dict:
             for result in response_dict['results']:
                 result['source_tool'] = 'tavily_search' # Ensure tool name is added
        return response_dict
    except Exception as e:
        logger.error(f"Error during Tavily search for query '{query}': {e}")
        return {"error": f"Tavily search failed: {e}"}

@tool(args_schema=FirecrawlInput)
def firecrawl_scrape_tool(url: str) -> Dict[str, Any]:
    """Retrieves the main text content of a single web page URL in Markdown format using Firecrawl.
    Use this when a search result snippet (from Tavily, etc.) is insufficient, and you need the full text content of a specific, highly relevant page.
    Provides clean Markdown suitable for analysis.
    """
    logger.info(f"Attempting to scrape URL with Firecrawl: {url}")
    try:
        firecrawl_client = config.get_firecrawl_client() # Raises error if not configured
    except Exception as e:
        logger.error(f"Failed to initialize Firecrawl client: {e}")
        return {"url": url, "content": None, "error": f"Firecrawl client initialization failed: {e}", "source_tool": "firecrawl_scrape_tool"}

    try:
        # Scrape the URL. Pass options within the 'params' dictionary.
        # Removed 'formats': ['markdown'] as it might be implicit or causing issues.
        scrape_params = {
            'pageOptions': {
                'onlyMainContent': True # Try to get only main content
            }
            # Removed 'formats': ['markdown']
        }
        response = firecrawl_client.scrape_url(url=url, params=scrape_params)

        # Check if response is the expected dictionary format
        if isinstance(response, dict) and 'markdown' in response:
            logger.info(f"Successfully scraped URL: {url}")
            # Add source tool info
            response['source_tool'] = 'firecrawl_scrape_tool'
            # Return the relevant parts (or the whole dict)
            # Ensure content is returned, even if None/empty, along with metadata
            return {
                "url": url,
                "markdown_content": response.get('markdown'),
                "metadata": response.get('metadata'), # Include metadata if available
                "source_tool": "firecrawl_scrape_tool"
            }
        elif isinstance(response, dict) and 'error' in response:
             logger.error(f"Firecrawl returned an error for {url}: {response['error']}")
             return {"url": url, "content": None, "error": response['error'], "source_tool": "firecrawl_scrape_tool"}
        else:
            # Handle unexpected response format from Firecrawl library
             logger.warning(f"Unexpected response format from Firecrawl for {url}. Type: {type(response)}, Content: {str(response)[:200]}...")
             return {"url": url, "content": None, "error": "Unexpected response format from Firecrawl.", "source_tool": "firecrawl_scrape_tool"}

    except requests.exceptions.HTTPError as http_err:
        # Catch HTTP errors specifically to log status code and response text if possible
        response_text = http_err.response.text if http_err.response else "No response body"
        logger.error(f"HTTPError during Firecrawl scraping for URL {url}: {http_err.response.status_code} {http_err}. Response: {response_text}", exc_info=False) # Keep traceback short
        return {"url": url, "content": None, "error": f"Firecrawl HTTP Error {http_err.response.status_code}: {response_text}", "source_tool": "firecrawl_scrape_tool"}
    except Exception as e:
        logger.error(f"Error during Firecrawl scraping for URL {url}: {e}", exc_info=True)
        return {"url": url, "content": None, "error": f"Firecrawl scraping failed: {e}", "source_tool": "firecrawl_scrape_tool"}

@tool(args_schema=NewsSearchInput)
def news_search(query: str, language: str = 'en', page_size: int = 10) -> Dict[str, Any]:
    """
    Searches recent news articles using NewsAPI. Useful for finding current events, latest developments, or official announcements related to a query.
    Returns a list of articles with titles, descriptions, URLs, and publication dates.
    """
    try:
        newsapi = config.get_news_api_client() # Get the initialized client
        if not newsapi: # The config function raises ValueError if key is missing, but check anyway
             logger.error("NewsAPI client could not be initialized (check config and NEWS_API_KEY).")
             return {"error": "NewsAPI client unavailable."}
        
        logger.info(f"Performing NewsAPI search for query: '{query}' (lang: {language}, size: {page_size})")
        # Use the client directly
        response = newsapi.get_everything(q=query, language=language, sort_by='relevancy', page_size=page_size)
        response['source_tool'] = 'news_search' # Add tool name
        logger.info(f"NewsAPI search completed. Status: {response.get('status')}. Found {len(response.get('articles', []))} articles.")
        return response
    except ValueError as ve: # Catch specific error from config if key is missing
        logger.error(f"Error initializing NewsAPI client: {ve}")
        return {"error": f"NewsAPI configuration error: {ve}"}
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}", exc_info=True)
        return {"error": f"NewsAPI search failed: {e}"}

@tool(args_schema=SearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo. Provides concise results with snippets and links.
    Good alternative or supplement to Tavily for general web searching.
    """
    # Use the underlying library directly for more control over output
    from duckduckgo_search import DDGS

    try:
        logger.info(f"Performing DuckDuckGo search for query: '{query}' with max_results={max_results}")
        # Use ddgs.text() which returns a list of dicts: {'title', 'href', 'body'}
        with DDGS() as ddgs:
            search_results = list(ddgs.text(query, max_results=max_results))

        processed_results = []
        if search_results:
            for res in search_results:
                processed_results.append({
                    'title': res.get('title'),
                    'url': res.get('href'), # Map href to url
                    'snippet': res.get('body'), # Map body to snippet
                    'source_tool': 'duckduckgo_search'
                })
        logger.info(f"DuckDuckGo search completed. Found {len(processed_results)} results.")
        return processed_results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}", exc_info=True)
        # Return error in the expected list format
        return [{'error': f"DuckDuckGo search failed: {e}", 'source_tool': 'duckduckgo_search'}]

@tool(args_schema=WikidataInput)
def wikidata_entity_search(query: str) -> Dict[str, Any]:
    """
    Queries Wikidata for structured information about specific entities (people, places, organizations, concepts).
    Use this when you need specific factual data points (e.g., birth date, capital city, official website) that are likely in Wikidata.
    Input should be the entity name.
    """
    wikidata_api = WikidataAPIWrapper()
    wikidata_tool = WikidataQueryRun(api_wrapper=wikidata_api)
    try:
        logger.info(f"Performing Wikidata query for: '{query}'")
        response = wikidata_tool.run(query)
        # Attempt to wrap in a dict for consistency, assuming response is often a string
        if isinstance(response, str):
            response_dict = {"result": response, "source_tool": "wikidata_entity_search"}
        elif isinstance(response, dict):
            response['source_tool'] = 'wikidata_entity_search'
            response_dict = response
        else:
            response_dict = {"result": str(response), "source_tool": "wikidata_entity_search"}
        logger.info(f"Wikidata query completed for: '{query}'")
        return response_dict
    except Exception as e:
        logger.error(f"Error during Wikidata query for '{query}': {e}", exc_info=True)
        return {"error": f"Wikidata query failed: {e}"}

# --- Gemini Google Search Tool (Updated context) ---
class GeminiGoogleSearchInput(BaseModel):
    query: str = Field(description="The query or sub-topic to investigate using Google Search synthesized by Gemini.")

def _gemini_google_search_and_parse_internal(query: str) -> Dict[str, Any]:
    """Internal function to call Gemini and attempt parsing."""
    gemini_llm = config.get_gemini_llm()
    if not gemini_llm:
        return {"error": "Gemini LLM not configured."}

    logger.info(f"Performing Gemini Google Search for query: '{query}'")
    try:
        # Assuming gemini_llm object handles the search-enabled call directly
        # The exact invocation might depend on how config.get_gemini_llm() sets it up
        # This might involve specific parameters or methods on the gemini_llm object.
        # Example: result = gemini_llm.invoke(query, search_enabled=True)
        # For now, using a placeholder invoke call:
        response = gemini_llm.invoke(query) # Adjust if invoke needs specific args for search

        # Extract raw text content (assuming response is a message object)
        raw_output = getattr(response, 'content', str(response))
        logger.info("Gemini search completed. Attempting to parse output.")

        # Call the parsing helper function (which now takes 'query')
        parsed_output = parse_gemini_output_with_llm(raw_output, query)

        if parsed_output:
            output_dict = parsed_output.dict()
            output_dict['source_tool'] = 'gemini_google_search_tool'
            return output_dict
        else:
            logger.warning("Failed to parse Gemini output, returning raw output.")
            return {"raw_output": raw_output, "error": "Parsing failed", "source_tool": "gemini_google_search_tool"}

    except Exception as e:
        logger.error(f"Error during Gemini Google Search for query '{query}': {e}", exc_info=True)
        return {"error": f"Gemini Google Search failed: {e}"}

@tool(args_schema=GeminiGoogleSearchInput)
def gemini_google_search_tool(query: str) -> Dict[str, Any]:
    """
    Uses Google Search via Gemini to answer a query, providing a synthesized answer with citations.
    Useful for direct questions or getting a summarized perspective with source links integrated.
    The output includes a summary, key facts, and source URLs.
    """
    return _gemini_google_search_and_parse_internal(query)

# --- FINISH Tool Schema (for binding, not execution) ---
class FinishSchema(BaseModel):
    reason: str = Field(default="Research complete. Sufficient information gathered.", description="A brief reason why the research is being concluded (e.g., 'sufficient evidence found', 'conflicting evidence found', 'scope covered').")

@tool(args_schema=FinishSchema)
def FINISH(reason: str) -> Dict[str, Any]:
    """
    Signals the end of the research process. When the agent calls this tool, the routing logic will handle generating the final report.
    """
    logger.info(f"Agent signaled FINISH. Reason: {reason}")
    return {"reason": reason}

# --- Tool Creation Function ---
def create_agent_tools(cfg: Optional[Dict] = None) -> List[BaseTool]:
    """Creates a list of available tools based on the provided configuration."""
    if cfg is None:
        cfg = config.load_config() # Load default config if none provided

    available_tools: List[BaseTool] = []

    # Query Decomposition Tool (Always include?)
    available_tools.append(QueryDecompositionTool())

    if cfg.get("enable_tavily", True):
        if config.TAVILY_API_KEY:
            available_tools.append(tavily_search)
        else:
            logger.warning("Tavily enabled but API key not found. Skipping Tavily tool.")

    if cfg.get("enable_duckduckgo", True):
        available_tools.append(duckduckgo_search)

    if cfg.get("enable_news_search", True):
        if config.NEWS_API_KEY:
             available_tools.append(news_search)
        else:
             logger.warning("News Search enabled but API key not found. Skipping News tool.")

    if cfg.get("enable_wikidata", False): # Defaulting Wikidata to False unless needed
        available_tools.append(wikidata_entity_search)

    if cfg.get("enable_firecrawl_scraper", True): # New config option
        if config.FIRECRAWL_API_KEY:
            available_tools.append(firecrawl_scrape_tool) # Add new tool
        else:
             logger.warning("Firecrawl Scraper enabled but API key not found. Skipping Firecrawl tool.")

    if cfg.get("enable_gemini_search", True):
        if config.GEMINI_API_KEY:
            available_tools.append(gemini_google_search_tool)
        else:
             logger.warning("Gemini Search enabled but API key not found. Skipping Gemini tool.")

    logger.info(f"Created agent tools: {[tool.name for tool in available_tools]}")
    # Include the FINISH tool to allow proper binding and recognition
    available_tools.append(FINISH)
    return available_tools