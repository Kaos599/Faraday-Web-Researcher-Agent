# Agent Structure

The Faraday Web Research Agent utilizes an agentic architecture built with LangGraph, running directly within the Streamlit application. This approach allows for dynamic, stateful execution of the research process rather than relying on a fixed, predefined sequence of steps.

## Core Components

1.  **LangGraph StateGraph**: The heart of the agent is a `StateGraph` defined in `research_system/agent.py`. This graph manages the flow of execution and maintains the research state.
    *   **State Schema**: The `ResearchState` TypedDict (defined in `research_system/schemas.py`) holds all the information accumulated during the research, including the original query, decomposed sub-queries, gathered evidence (snippets, links, summaries from tools), the final report, and any error messages.
    *   **Nodes**: The graph consists of several key nodes:
        *   `start_research`: Initializes the state with the user's query.
        *   `plan_initial_research` / `analyze_initial_results`: The agent reflects on the query and initial findings to decide the next steps. (Implicit within the main agent logic)
        *   `execute_tool`: Invokes the chosen research tool (e.g., Tavily search, Firecrawl scrape).
        *   `process_tool_result`: Takes the raw output from a tool, formats it, and adds it to the `evidence` list in the state.
        *   `should_continue`: A conditional edge logic point. The agent (LLM) decides whether more research is needed or if it has enough information to finish.
        *   `generate_report`: If the agent decides to finish, this node calls the LLM with the synthesis prompt (`REPORT_SYNTHESIS_TEMPLATE` from `research_system/prompts.py`) to generate the final `ResearchReport`.
        *   `handle_error`: Captures and formats errors encountered during the process.
    *   **Edges**: Edges connect the nodes, defining the possible paths of execution.
        *   `START` -> `agent` (main agent execution logic node)
        *   `agent` -> `execute_tool`: When the agent decides to use a tool.
        *   `execute_tool` -> `process_tool_result`: After a tool runs successfully.
        *   `process_tool_result` -> `should_continue`: After processing results, check if more work is needed.
        *   `should_continue` -> `agent`: If more research is needed.
        *   `should_continue` -> `generate_report`: If research is complete.
        *   `generate_report` -> `END`: The final step.
        *   Edges also lead to `handle_error` from various points where failures can occur.

2.  **Primary LLM (Agent Executor)**: A Large Language Model (configured in `research_system/config.py` and used in `research_system/agent.py`) acts as the "brain" of the agent. It takes the current state and the `AGENT_SYSTEM_PROMPT` to:
    *   Analyze the query and research progress.
    *   Plan the next steps.
    *   Select the most appropriate tool for the task.
    *   Decide when the research is sufficient.

3.  **Research Tools**: Defined in `research_system/tools.py`, these are the functions the agent can call to interact with the external world (web search, scraping, etc.). Each tool is specifically designed to perform a distinct research task.

4.  **Streamlit Interface (`app.py`)**: Provides the user interface for inputting queries and displaying the final report or errors. It directly calls the LangGraph agent's `run_web_research` function.

## Execution Flow

The process is cyclical and driven by the agent's decisions:

1.  The Streamlit app triggers the `run_web_research` function in `agent.py`.
2.  The `StateGraph` is invoked.
3.  The `agent` node (using the Primary LLM) analyzes the state and decides on an action (e.g., call `tavily_search`).
4.  The graph transitions to the `execute_tool` node, calling the chosen tool function.
5.  The tool function (`research_system/tools.py`) executes, interacting with external APIs/web.
6.  The result returns to the `process_tool_result` node, updating the graph's state.
7.  The `should_continue` conditional edge checks if the agent wants to finish.
8.  If continuing, the graph loops back to the `agent` node for the next decision.
9.  If finishing, the graph transitions to `generate_report`.
10. The final report (or error) is returned to the Streamlit app.

This structure allows for flexibility, enabling the agent to adapt its strategy based on the information it gathers, rather than following a rigid script. 