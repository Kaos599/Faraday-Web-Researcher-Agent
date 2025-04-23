"""
This module contains wrapper functions for various search and information gathering tools
used by the Web Research Agent system, refactored as Langchain Tools.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Type
import requests
from . import config

from .schemas import ResearchReport, Source, IntermediateStep, ResearchRequest, ErrorResponse, GeminiParsedOutput
from .prompts import QUERY_DECOMPOSITION_TEMPLATE

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import tool, BaseTool
from .config import get_primary_llm, get_azure_openai_parser_llm
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikidata import WikidataAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from newsapi import NewsApiClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Internal Helper Functions ---
def parse_gemini_output_with_llm(gemini_raw_output: str, query: str) -> Optional[GeminiParsedOutput]:
    """
    Uses the secondary Azure OpenAI LLM to parse the raw text output from the
    Gemini+GoogleSearch call into a structured format.

    Args:
        gemini_raw_output: The raw string output from perform_gemini_google_search.
        query: The original research query, provided for context.

    Returns:
        A GeminiParsedOutput Pydantic object, or None if parsing fails or LLM is unavailable.
    """
    parser_llm = get_azure_openai_parser_llm()
    if not parser_llm:
        logger.error("Azure OpenAI Parser LLM is not available for Gemini parsing.")
        return None

    parser = JsonOutputParser(pydantic_object=GeminiParsedOutput)

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
        parsed_result_dict = chain.invoke({
            "query": query,
            "gemini_raw_output": gemini_raw_output,
            "format_instructions": parser.get_format_instructions()
        })
        logger.info("Successfully parsed Gemini output. Filtering sources...")

        filtered_sources = []
        if "sources" in parsed_result_dict and isinstance(parsed_result_dict["sources"], list):
            for source_url in parsed_result_dict["sources"]:
                if isinstance(source_url, str) and source_url.strip().lower().startswith(("http://", "https://")):
                    filtered_sources.append(source_url.strip())
                else:
                    logger.warning(f"Filtering out invalid source URL from Gemini output: {source_url}")
            parsed_result_dict["sources"] = filtered_sources
        else:
             logger.warning("No 'sources' list found in parsed Gemini output or it's not a list.")
             parsed_result_dict["sources"] = []

        return GeminiParsedOutput(**parsed_result_dict)
    except OutputParserException as ope:
         logger.error(f"Failed to parse Gemini output. Parser Error: {ope}. Raw output was:\n{gemini_raw_output[:500]}...")
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


# --- Query Decomposition Tool ---
class QueryDecompositionInput(BaseModel):
    query: str = Field(..., description="The complex research query to decompose into sub-topics or questions.")

class QueryDecompositionOutput(BaseModel):
    sub_queries: List[str] = Field(..., description="A list of simpler, searchable sub-queries or sub-topics derived from the original query.")


class QueryDecompositionTool(BaseTool):
    """Tool to decompose complex research queries."""
    name: str = "query_decomposition_tool"
    description: str = (
        "Decomposes a complex research query (containing multiple facets, entities, or questions) into a list of simpler, "
        "focused sub-queries or sub-topics. Use this *first* on complex queries to guide the research strategy and generate targeted searches for other tools."
    )
    args_schema: Type[BaseModel] = QueryDecompositionInput

    def _run(self, query: str) -> Dict[str, Any]:
        """Executes the query decomposition."""
        logger.info(f"Executing Query Decomposition Tool for query: '{query}'")
        llm = get_primary_llm()
        if not llm:
            logger.error("Primary LLM not available for Query Decomposition Tool.")
            return {"error": "LLM unavailable for decomposition."}

        from langchain_core.output_parsers import StrOutputParser
        import ast
        prompt = QUERY_DECOMPOSITION_TEMPLATE
        chain = prompt | llm | StrOutputParser()

        try:
            result_str = chain.invoke({"query": query})
            try:
                sub_queries_list = ast.literal_eval(result_str.strip())
                if not isinstance(sub_queries_list, list):
                    raise ValueError("Parsed result is not a list")
                result = {"sub_queries": sub_queries_list}
                logger.info(f"Query decomposition successful. Found {len(result.get('sub_queries', []))} sub-queries.")
            except (SyntaxError, ValueError, TypeError) as parse_err:
                logger.error(f"Failed to parse sub-query list string: {parse_err}. Raw result: {result_str}")
                result = {"sub_queries": [result_str]}
            return result
        except Exception as e:
            logger.error(f"Error during query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Query decomposition failed: {str(e)}"}

    async def _arun(self, query: str) -> Dict[str, Any]:
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
                result = {"sub_queries": [result_str]}
            return result
        except Exception as e:
            logger.error(f"Error during async query decomposition for '{query}': {e}", exc_info=True)
            return {"error": f"Async query decomposition failed: {str(e)}"}

# --- Search & Retrieval Tools ---

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
                 result['source_tool'] = 'tavily_search'
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
    firecrawl_client = config.get_firecrawl_client()
    if not firecrawl_client:
        logger.error("Firecrawl client is not available.")
        return {"error": "Firecrawl client unavailable."}
    try:
        # Firecrawl returns a list of pages, even for single URLs. Take the first result.
        result = firecrawl_client.scrape_url(url)
        if result and isinstance(result, list) and result[0]:
             scraped_data = result[0]
             # Add source_tool field
             scraped_data['source_tool'] = 'firecrawl_scrape_tool'
             logger.info(f"Successfully scraped {url}")
             return scraped_data # Return the dict
        else:
             logger.warning(f"Firecrawl scrape returned empty result for {url}")
             return {"url": url, "markdown_content": "", "metadata": {}, "source_tool": 'firecrawl_scrape_tool', "error": "Empty scrape result"}
    except Exception as e:
        logger.error(f"Error during Firecrawl scrape for URL {url}: {e}")
        return {"url": url, "markdown_content": "", "metadata": {}, "source_tool": 'firecrawl_scrape_tool', "error": f"Firecrawl scrape failed: {e}"}

@tool(args_schema=NewsSearchInput)
def news_search(query: str, language: str = 'en', page_size: int = 10) -> Dict[str, Any]:
    """
    Searches recent news articles for a given query using NewsAPI.
    Useful for finding timely information and current events related to the research topic.
    """
    logger.info(f"Performing NewsAPI search for query: '{query}'")
    newsapi_client = config.get_newsapi_client()
    if not newsapi_client:
        logger.error("NewsAPI client is not available.")
        return {"error": "NewsAPI client unavailable."}
    try:
        # NewsAPI returns a dict with 'articles' key
        response = newsapi_client.get_everything(q=query, language=language, sort_by='relevancy', page_size=page_size)
        if response and response.get('articles'):
             for article in response['articles']:
                 article['source_tool'] = 'news_search' # Add tool name
             logger.info(f"NewsAPI search completed. Found {len(response['articles'])} articles.")
             return {"articles": response['articles']}
        elif response and response.get('totalResults') == 0:
             logger.info(f"NewsAPI search found 0 articles for '{query}'.")
             return {"articles": [], "message": "No articles found."}
        else:
             logger.warning(f"NewsAPI search returned unexpected response structure for '{query}': {response}")
             return {"articles": [], "error": "Unexpected API response"}
    except Exception as e:
        logger.error(f"Error during NewsAPI search for query '{query}': {e}")
        return {"articles": [], "error": f"NewsAPI search failed: {e}"}

@tool(args_schema=SearchInput)
def duckduckgo_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a simple web search using DuckDuckGo.
    Provides concise results quickly, good for finding definitions or quick facts.
    Returns a list of search results with titles, URLs, and snippets.
    """
    logger.info(f"Performing DuckDuckGo search for query: '{query}'")
    try:
        # DuckDuckGoSearchRun directly returns a list of dicts
        ddg_search = DuckDuckGoSearchAPIWrapper(region="us-en", max_results=max_results)
        results = ddg_search.results(query, max_results=max_results)
        # Add source_tool to each result
        for result in results:
            result['source_tool'] = 'duckduckgo_search'
        logger.info(f"DuckDuckGo search completed. Found {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for query '{query}': {e}")
        return {"error": f"DuckDuckGo search failed: {e}"}

@tool(args_schema=WikidataInput)
def wikidata_entity_search(query: str) -> Dict[str, Any]:
    """
    Searches Wikidata for structured data about a specific entity.
    Useful for retrieving factual attributes (dates, relationships, properties) for well-known people, places, or concepts.
    Input should be the entity name (e.g., 'Albert Einstein', 'Eiffel Tower').
    """
    logger.info(f"Performing Wikidata search for entity: '{query}'")
    try:
        wikidata_wrapper = WikidataAPIWrapper()
        tool = WikidataQueryRun(api_wrapper=wikidata_wrapper)
        # The run method returns a string, need to parse it potentially
        result_str = tool.run(query)
        # Simple approach: Wrap the string result in a dict with tool name
        logger.info(f"Wikidata search completed for '{query}'.")
        return {"query": query, "result": result_str, "source_tool": 'wikidata_entity_search'}
    except Exception as e:
        logger.error(f"Error during Wikidata search for entity '{query}': {e}")
        return {"error": f"Wikidata search failed: {e}", "query": query, "result": None, "source_tool": 'wikidata_entity_search'}

# --- Gemini Google Search Tool ---
# This tool wraps a specific capability (e.g., a Gemini model with Google Search enabled)
# It's expected to return a specific format that needs parsing.

class GeminiGoogleSearchInput(BaseModel):
    query: str = Field(description="The query or sub-topic to investigate using Google Search synthesized by Gemini.")

def _gemini_google_search_and_parse_internal(query: str) -> Dict[str, Any]:
    """
    Internal function to call a Gemini model with Google Search and parse its output.
    This assumes an underlying model that can perform search and synthesize.
    The actual implementation depends on the specific LLM provider and setup.
    Returns parsed structured data or an error.
    """
    logger.info(f"Calling internal Gemini Google Search for query: '{query}'")
    # This is a placeholder for the actual LLM call with search capabilities
    # In a real implementation, this would involve calling the Gemini API
    # with the appropriate search parameters or using a specific LangChain integration.
    # For demonstration, we'll return a dummy structure or integrate with a real client if available.

    gemini_search_llm = config.get_gemini_google_search_llm() # Assuming a separate config entry for this

    if not gemini_search_llm:
         logger.error("Gemini Google Search LLM is not configured.")
         return {"error": "Gemini Google Search LLM not available.", "query": query}

    try:
        # This is where you would invoke the LLM capable of search
        # Example (replace with actual LLM invocation logic):
        # raw_output = gemini_search_llm.invoke(f"Search for: {query} and summarize findings including sources.")

        # *** Placeholder: Replace with actual LLM call ***
        raw_output = f"Simulated search result for '{query}'. Key finding: relevant info. Sources: http://example.com/simulated http://anothersite.org/data"
        # *** End Placeholder ***

        logger.info(f"Raw output from Gemini Search LLM: {raw_output[:200]}...")

        # Use the parsing helper
        parsed_data = parse_gemini_output_with_llm(raw_output, query)

        if parsed_data:
             # Convert Pydantic object to dict and add source_tool
             parsed_dict = parsed_data.model_dump()
             parsed_dict['source_tool'] = 'gemini_google_search_tool'
             logger.info(f"Successfully parsed Gemini Google Search output for '{query}'.")
             return parsed_dict
        else:
             logger.error(f"Failed to parse Gemini Google Search output for '{query}'.")
             return {"error": "Failed to parse Gemini Google Search output.", "query": query, "raw_output": raw_output[:500] + "..."}

    except Exception as e:
        logger.error(f"Error during Gemini Google Search for query '{query}': {e}")
        return {"error": f"Gemini Google Search failed: {e}", "query": query}


@tool(args_schema=GeminiGoogleSearchInput)
def gemini_google_search_tool(query: str) -> Dict[str, Any]:
    """
    Leverages a Google Search-enabled Gemini model to find and synthesize information for a query.
    Excellent for getting a quick, synthesized overview and relevant sources directly from the LLM.
    """
    return _gemini_google_search_and_parse_internal(query)

# --- FINISH Tool ---
# This is a special tool to signal the agent has completed its task.

class FinishSchema(BaseModel):
    reason: str = Field(default="Research complete. Sufficient information gathered.", description="A brief reason why the research is being concluded (e.g., 'sufficient evidence found', 'conflicting evidence found', 'scope covered').")

@tool(args_schema=FinishSchema)
def FINISH(reason: str) -> Dict[str, Any]:
    """
    Signals that the agent has completed its research task and is ready to return a final answer.
    Use this tool when you have gathered sufficient information to synthesize a comprehensive report.
    Provide a brief reason for finishing.
    """
    logger.info(f"FINISH tool called with reason: {reason}")
    # This tool doesn't perform an action that returns data for the agent loop
    # Its purpose is to route the graph to the final synthesis step.
    # We return a simple confirmation.
    return {"status": "FINISH signal received", "reason": reason}

# --- Tool Creation Factory ---
def create_agent_tools(cfg: Optional[Dict] = None) -> List[BaseTool]:
    """
    Creates and returns a list of initialized tools for the agent.
    Based on provided configuration.
    """
    logger.info("Creating agent tools...")
    tools: List[BaseTool] = [
        # Initialize standard tools
        # Remove FirecrawlInput as it's likely just a schema, not an executable tool
        # FirecrawlInput,
        tavily_search,
        firecrawl_scrape_tool,
        news_search,
        duckduckgo_search,
        wikidata_entity_search,
        QueryDecompositionTool(), # Query Decomposition is a class
        gemini_google_search_tool,
        FINISH # Include the FINISH tool
    ]
    logger.info(f"Created {len(tools)} tools.")
    return tools