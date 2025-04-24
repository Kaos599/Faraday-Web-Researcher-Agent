# Prompt Design Philosophy

The effectiveness of the Faraday Web Research Agent heavily relies on the design of its prompts, located in `research_system/prompts.py`. These prompts guide the Large Language Models (LLMs) used for reasoning, tool use, and final report generation.

## Key Prompts and Their Roles

1.  **`AGENT_SYSTEM_PROMPT`**: This is the master prompt for the core reasoning LLM.
    *   **Purpose**: To define the agent's persona, overall goal, available tools, and the expected multi-phase research process.
    *   **Design Choices**:
        *   **Persona**: Clearly defines the agent as an "AI Web Research Agent".
        *   **Goal**: Explicitly states the objective: conduct thorough research and compile a comprehensive, unbiased report.
        *   **Tool Listing**: Dynamically injects descriptions of available tools (`{{tool_descriptions}}`), ensuring the agent knows its capabilities.
        *   **Phased Approach**: Breaks down the complex research task into logical phases (Analysis & Planning, Iterative Research, Finish & Report).
        *   **Tool Guidance**: Provides explicit instructions on *when* and *why* to use specific tools (e.g., diverse initial searches, using specific tools like `news_search` or `firecrawl_scrape_tool` for targeted tasks, emphasizing the use of multiple tools).
        *   **Mandatory Finish**: Clearly instructs the agent that its FINAL action *must* be to call the `FINISH` tool, preventing endless loops and ensuring the workflow concludes.
        *   **Emphasis on Process**: Encourages step-by-step thinking, justification of tool choices, objectivity, and source citation.

2.  **`REPORT_SYNTHESIS_TEMPLATE`**: Used by the LLM during the final report generation phase.
    *   **Purpose**: To instruct the LLM on how to synthesize the collected evidence into a structured `ResearchReport` JSON object.
    *   **Design Choices**:
        *   **Context**: Provides the original user query (`{query}`) and all gathered evidence (`{formatted_evidence}`) as context.
        *   **Clear Instructions**: Outlines the steps for synthesis: review the query, analyze evidence, structure the report according to the schema, write objectively, cite sources, and acknowledge limitations.
        *   **Schema Enforcement**: Uses `PydanticOutputParser` to automatically include detailed formatting instructions (`{format_instructions}`) based on the `ResearchReport` Pydantic model (`research_system/schemas.py`). This significantly increases the likelihood of receiving correctly formatted JSON.
        *   **Strict Output Format**: Explicitly demands *only* the JSON output, minimizing extraneous text.

3.  **`QUERY_DECOMPOSITION_TEMPLATE`**: Used by the optional `query_decomposition_tool`.
    *   **Purpose**: To guide an LLM in breaking down a complex user query into smaller, manageable sub-queries.
    *   **Design Choices**:
        *   **Clear Task**: Instructs the LLM to identify key aspects and formulate concise sub-topics suitable for searching.
        *   **Examples**: Provides few-shot examples to demonstrate the desired input/output format and the style of decomposition.
        *   **Strict Output Format**: Requires the output to be *only* a Python list of strings, simplifying parsing in the tool.

4.  **`SEARCH_QUERY_GENERATION_TEMPLATE`** (Potential Use / Adaptation):
    *   **Purpose**: Could be used to generate varied search engine queries based on the main query and a sub-topic.
    *   **Design**: Focuses on generating 1-3 concise, effective queries suitable for standard search tools.

5.  **`GEMINI_OUTPUT_PARSER_TEMPLATE`** (If Gemini Tool is Heavily Used):
    *   **Purpose**: To reliably extract structured information (summary, facts, URLs) from the potentially verbose output of the Gemini-based search tool.
    *   **Design**: Provides context (original query), the raw Gemini output, and specific format instructions (likely using another Pydantic model and parser) to guide the LLM in extracting the desired fields into JSON.

## General Principles

*   **Clarity and Specificity**: Instructions are direct and unambiguous.
*   **Role Definition**: Prompts clearly define the role and goal for the LLM in that specific context.
*   **Context Provision**: Necessary information (query, evidence, tool descriptions) is provided.
*   **Structured Output**: Where possible, prompts leverage format instructions (often via Pydantic parsers) to ensure the LLM produces easily parsable, structured output (especially JSON).
*   **Iterative Refinement**: The prompts were likely developed through trial and error, refining instructions based on observed LLM behavior to achieve the desired agent performance. 