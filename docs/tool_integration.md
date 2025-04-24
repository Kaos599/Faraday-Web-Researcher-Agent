# Tool Integration

The agent interacts with the outside world through a set of defined tools, primarily located in `research_system/tools.py`. These tools encapsulate the logic for calling external APIs (like search engines, news APIs, knowledge bases) and processing their results.

## Tool Definition and Registration

1.  **Tool Implementation (`research_system/tools.py`)**: Each tool is implemented as a Python function.
    *   These functions handle:
        *   Taking necessary arguments (e.g., search query, URL).
        *   Calling the relevant external API or library (e.g., `TavilyClient`, `DuckDuckGoSearchRun`, `FirecrawlLoader`, `NewsApiClient`, `WikidataAPIWrapper`, Google `VertexAI`).
        *   Performing basic processing or formatting of the raw results.
        *   Handling potential exceptions during the API call (often wrapped in `try...except` blocks).
    *   Examples include `tavily_search`, `duckduckgo_search`, `firecrawl_scrape_tool`, `news_search`, `wikidata_entity_search`, and `gemini_google_search_tool`.
    *   An internal `query_decomposition_tool` also exists, using an LLM to break down queries.

2.  **Tool Schema (`research_system/schemas.py`)**: Pydantic models (like `SearchToolInput`, `ScrapeToolInput`) are often used to define the expected input arguments for each tool. This helps with validation and provides structure.

3.  **Tool Representation for Agent**: Each tool function needs to be represented in a way the LangGraph agent and the LLM can understand.
    *   **Langchain Tools**: The functions are typically wrapped using LangChain's `@tool` decorator or converted into `Tool` objects. This automatically extracts the function signature and docstring to inform the LLM about the tool's purpose and arguments.
    *   **Tool List**: A list of these `Tool` objects is assembled (often in `research_system/agent.py` or `research_system/config.py`).

## Tool Selection and Invocation

1.  **Agent Prompting**: The `AGENT_SYSTEM_PROMPT` in `research_system/prompts.py` includes descriptions of the available tools. These descriptions are crucial for the LLM to understand *what* each tool does and *when* it might be appropriate to use it.

2.  **LLM Decision**: During its reasoning cycle, the primary LLM analyzes the current research state and the goal. Based on the prompt and its understanding of the tools, it decides which tool (if any) to call next and with what arguments.

3.  **LangGraph Execution**: The LangGraph framework receives the LLM's decision (formatted as a tool call).
    *   The `execute_tool` node (or similar logic within the agent) identifies the requested tool function.
    *   It parses the arguments provided by the LLM.
    *   It calls the corresponding Python function from `research_system/tools.py` with those arguments.

4.  **Observation Processing**: The output (return value) from the tool function is captured as an "observation".
    *   The `process_tool_result` node takes this observation.
    *   It typically formats the observation into a standardized structure (like adding it to the `evidence` list in the `ResearchState` along with metadata like the tool used and source URL/title).
    *   This updated state is then fed back to the LLM for the next reasoning step.

## Specific Tool Examples

*   **Search Tools (`tavily_search`, `duckduckgo_search`, `gemini_google_search_tool`)**: Take a query string, call the respective search API, and return a list of search results (snippets, links, titles). Gemini might return a more summarized answer with citations.
*   **Scraping (`firecrawl_scrape_tool`)**: Takes a specific URL, uses the Firecrawl service to fetch and parse the main content of that page, returning it usually in Markdown format.
*   **News (`news_search`)**: Takes a query, calls the NewsAPI, and returns recent relevant news articles.
*   **Knowledge Base (`wikidata_entity_search`)**: Takes an entity name, queries Wikidata, and returns structured factual information about that entity.
*   **Internal (`query_decomposition_tool`)**: Takes the main query, uses an LLM with a specific prompt (`QUERY_DECOMPOSITION_TEMPLATE`), and returns a list of sub-queries.

This integration allows the agent to dynamically leverage external information sources as needed to fulfill the research query. 