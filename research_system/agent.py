import operator
import logging
import uuid
import json
from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from pydantic import BaseModel, Field

from .config import get_primary_llm
from .tools import create_agent_tools
from .prompts import AGENT_SYSTEM_PROMPT, REPORT_SYNTHESIS_TEMPLATE, report_parser
from .schemas import ResearchReport, Source, IntermediateStep, ResearchRequest, ErrorResponse
from langchain_core.exceptions import OutputParserException

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Represents the state of the Web Research Agent."""
    query: str # Renamed from claim
    research_id: str # Renamed from claim_id
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Keep intermediate_steps as tuple for now, synthesis node will parse
    intermediate_steps: Annotated[List[tuple[ToolInvocation, Any]], operator.add] = []
    # Holds the final structured research report
    final_result: Optional[ResearchReport] = None
    # Optional field to track collected sources explicitly if needed beyond intermediate_steps
    # collected_sources: List[Source] = [] # Could add this if parsing sources from steps becomes too complex


# --- Agent Nodes ---

def agent_node(state: AgentState, agent, tools, name: str):
    """Node that calls the agent model to decide the next action."""
    logger.info(f"[{name} - ID: {state.get('research_id')}] Agent node executing.")
    # Prepare system prompt using the new AGENT_SYSTEM_PROMPT
    system_prompt_content = AGENT_SYSTEM_PROMPT.format(
        tool_descriptions="\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    )

    # Prepend the system message to the current message history from the state
    messages_for_llm = [SystemMessage(content=system_prompt_content)] + list(state["messages"])

    logger.info(f"[{name} - ID: {state.get('research_id')}] Invoking agent LLM with {len(messages_for_llm)} total messages...")

    # Invoke the agent
    result: BaseMessage = agent.invoke(messages_for_llm)

    logger.info(f"[{name} - ID: {state.get('research_id')}] Agent LLM raw response type: {type(result)}")
    # Log content carefully, might be large
    # logger.debug(f"[{name}] Agent LLM raw response content: {result.content[:500]}...")

    output_messages = [result] if isinstance(result, BaseMessage) else []
    if not output_messages:
         logger.warning(f"[{name} - ID: {state.get('research_id')}] Agent LLM did not return a valid BaseMessage. Result: {result}")

    return {"messages": output_messages}


def tool_node(state: AgentState, tool_executor, name: str):
    """Node that executes the tool chosen by the agent."""
    logger.info(f"[{name} - ID: {state.get('research_id')}] Tool node executing.")
    messages = state["messages"]
    if not messages:
        logger.warning(f"[{name} - ID: {state.get('research_id')}] No messages found in state. Cannot execute tool.")
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    last_message = messages[-1]

    # Check if the last message is an AIMessage with tool_calls
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        # If not, it might be the FINISH signal or an error, handle in should_continue
        logger.info(f"[{name} - ID: {state.get('research_id')}] Last message is not an AIMessage with tool calls. Type: {type(last_message)}. Content: {getattr(last_message, 'content', '')[:100]}...")
        # Pass the state through without adding tool messages
        return {"messages": [], "intermediate_steps": state.get("intermediate_steps", [])}

    tool_invocation_list = []
    tool_messages = []
    intermediate_steps_updates = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.get("name")
        tool_input = tool_call.get("args", {})
        tool_call_id = tool_call.get('id')

        if not tool_name or not tool_call_id:
            logger.warning(f"[{name} - ID: {state.get('research_id')}] Skipping invalid tool call in message: {tool_call}")
            continue

        # Handle the FINISH signal - it shouldn't be executed as a tool here
        # The routing logic in should_continue handles FINISH.
        # If the agent mistakenly calls FINISH as a tool, log a warning.
        if tool_name.upper() == "FINISH":
            logger.warning(f"[{name} - ID: {state.get('research_id')}] Agent attempted to call FINISH as a tool. This should be handled by routing logic. Skipping tool execution for FINISH call.")
            continue

        logger.info(f"[{name} - ID: {state.get('research_id')}] Preparing to execute tool: {tool_name} (Call ID: {tool_call_id}) with input: {tool_input}")

        tool_invocation = ToolInvocation(tool=tool_name, tool_input=tool_input)
        tool_invocation_list.append(tool_invocation)

        try:
            # Execute the tool
            response = tool_executor.invoke(tool_invocation)
            logger.info(f"[{name} - ID: {state.get('research_id')}] Tool '{tool_name}' execution completed. Response type: {type(response)}")

            # Standardize observation format (prefer string for LLM consumption)
            if isinstance(response, (str, dict, list)):
                 observation = json.dumps(response) if isinstance(response, (dict, list)) else response
            elif isinstance(response, ToolMessage): # Should ideally not happen with standard executors
                 observation = response.content
            else:
                 observation = str(response)

            intermediate_steps_updates.append((tool_invocation, observation))
            tool_messages.append(ToolMessage(content=observation, tool_call_id=tool_call_id))

        except Exception as e:
            logger.error(f"[{name} - ID: {state.get('research_id')}] Error executing tool {tool_name} (Call ID: {tool_call_id}): {e}", exc_info=True)
            error_message = f"Error executing tool {tool_name}: {str(e)}"
            intermediate_steps_updates.append((tool_invocation, error_message))
            tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call_id))

    new_intermediate_steps = state.get("intermediate_steps", []) + intermediate_steps_updates

    return {"messages": tool_messages, "intermediate_steps": new_intermediate_steps}


# --- Node for Final Report Generation (Renamed and Rewritten) ---
def generate_final_report_node(state: AgentState, name: str) -> Dict[str, Optional[ResearchReport]]:
    """Generates the final structured research report by calling the synthesis LLM."""
    research_id = state.get('research_id')
    logger.info(f"[{name} - ID: {research_id}] Generating final research report.")

    synthesis_model = get_primary_llm()
    if not synthesis_model:
        logger.error(f"[{name} - ID: {research_id}] Primary LLM (for synthesis) not configured. Cannot generate final report.")
        # Return None or a minimal error report if desired
        return {"final_result": None}

    intermediate_steps = state.get("intermediate_steps", [])
    original_query = state.get("query", "No query provided")

    # --- Start: Source Extraction and Filtering ---
    formatted_evidence = ""
    potential_sources: List[Source] = [] # Store potential sources before filtering
    unique_source_urls = set()

    logger.info(f"[{name} - ID: {research_id}] Formatting evidence and extracting sources from {len(intermediate_steps)} intermediate steps.")
    for step_index, (tool_invocation, observation) in enumerate(intermediate_steps):
        step_number = step_index + 1
        tool_name = tool_invocation.tool
        tool_input = tool_invocation.tool_input
        formatted_evidence += f"\n--- Step {step_number}: Tool Used: {tool_name} ---\n"
        formatted_evidence += f"Input: {str(tool_input)[:200]}{'...' if len(str(tool_input)) > 200 else ''}\n"
        formatted_evidence += f"Observation:\n{str(observation)[:1000]}{'...' if len(str(observation)) > 1000 else ''}\n"

        # Enhanced source extraction (Add potential sources to list)
        try:
            obs_data = json.loads(observation) if isinstance(observation, str) else observation
            if isinstance(obs_data, dict):
                # Tavily/NewsAPI/Standard Search
                results = obs_data.get('results') or obs_data.get('articles')
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict):
                            url = item.get('url') or item.get('link')
                            title = item.get('title')
                            snippet = item.get('snippet') or item.get('content') or item.get('description')
                            if url: # Check if url exists
                                potential_sources.append(
                                    Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                                )

                # Gemini Search Parsed Output - Now expects filtered list from tool
                elif tool_name == 'gemini_google_search_tool' and 'sources' in obs_data:
                    urls = obs_data.get('sources', []) # This list should already be filtered in tools.py
                    gemini_snippet = obs_data.get('summary') or (obs_data.get('key_facts')[0] if obs_data.get('key_facts') else 'N/A')
                    for url in urls:
                        if url: # Double check url is not empty/None
                             potential_sources.append(
                                 Source(url=url, title=f"Source from Gemini Search for '{str(tool_input)[:50]}...'", snippet=gemini_snippet[:200], tool_used=tool_name)
                             )

                # Firecrawl Output
                elif tool_name == 'firecrawl_scrape_tool' and 'url' in obs_data and 'markdown_content' in obs_data:
                    url = obs_data.get('url')
                    if url:
                        title = obs_data.get('metadata', {}).get('title') or url
                        snippet = (obs_data.get('markdown_content') or '')[:200] + '...'
                        potential_sources.append(
                            Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                        )

                # DuckDuckGo Search Output
                elif tool_name == 'duckduckgo_search':
                     # ddg output is directly the list
                     if isinstance(obs_data, list):
                         for item in obs_data:
                             if isinstance(item, dict) and 'url' in item and 'title' in item:
                                 url = item.get('url')
                                 title = item.get('title')
                                 snippet = item.get('snippet') or item.get('body')
                                 if url:
                                     potential_sources.append(
                                         Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                                     )
                     else:
                          logger.warning(f"[{name} - ID: {research_id}] Unexpected format for DuckDuckGo output in step {step_number}: {type(obs_data)}")

        except json.JSONDecodeError:
            logger.warning(f"[{name} - ID: {research_id}] Observation for step {step_number} is not valid JSON. Skipping source extraction for this step. Content: {str(observation)[:100]}...")
        except Exception as e:
            logger.warning(f"[{name} - ID: {research_id}] Error processing sources from step {step_number} observation: {e}", exc_info=False)

    # --- Filter and Deduplicate Sources --- 
    filtered_sources: List[Source] = []
    for source in potential_sources:
        # Check if URL is not None, not empty, not 'N/A' (case-insensitive), and not already added
        if (
            source.url and 
            isinstance(source.url, str) and 
            source.url.strip() and 
            source.url.strip().lower() != 'n/a' and 
            source.url not in unique_source_urls
        ):
             # Basic check for valid URL start (can be enhanced)
             if source.url.strip().lower().startswith(('http://', 'https://')):
                 filtered_sources.append(source) # Add valid, unique source
                 unique_source_urls.add(source.url)
             else:
                  logger.warning(f"[{name} - ID: {research_id}] Filtering out source with invalid URL format: {source.url}")
        elif source.url in unique_source_urls:
             logger.debug(f"[{name} - ID: {research_id}] Skipping duplicate source URL: {source.url}")
        else:
             logger.warning(f"[{name} - ID: {research_id}] Filtering out source with invalid/missing URL: {source.url}")
    
    logger.info(f"[{name} - ID: {research_id}] Total unique and valid sources filtered for report: {len(filtered_sources)}")
    # --- End: Source Extraction and Filtering ---

    formatted_evidence += "\n-- End of Evidence --\n"

    # Prepare the prompt for the synthesis LLM
    # Note: The report_parser requires the sources list to be passed in the context
    # for validation if it refers to indices. We pass the *filtered* list.
    prompt_context = {
        "query": original_query,
        "formatted_evidence": formatted_evidence,
        "sources": filtered_sources # Pass the filtered list for the parser
    }
    # Use the imported prompt object directly
    synthesis_prompt_template = REPORT_SYNTHESIS_TEMPLATE 
    synthesis_chain = synthesis_prompt_template | synthesis_model | report_parser

    logger.info(f"[{name} - ID: {research_id}] Invoking synthesis LLM chain...")
    try:
        # Invoke the chain which includes the parser
        final_report_object: ResearchReport = synthesis_chain.invoke(prompt_context)
        logger.info(f"[{name} - ID: {research_id}] Synthesis and parsing successful.")
        # Ensure the filtered sources are part of the final report object
        # The parser should handle creating the object with the sources, but we can double-check
        if not final_report_object.sources:
             logger.warning(f"[{name} - ID: {research_id}] Final report object created but sources list is empty. Assigning filtered sources.")
             final_report_object.sources = filtered_sources
        elif len(final_report_object.sources) != len(filtered_sources):
             logger.warning(f"[{name} - ID: {research_id}] Mismatch between filtered sources ({len(filtered_sources)}) and sources in parsed report ({len(final_report_object.sources)}). Using parsed report sources.")
        
        # Assign the generated report to the state
        return {"final_result": final_report_object}

    except OutputParserException as ope:
        logger.error(f"[{name} - ID: {research_id}] Failed to invoke synthesis LLM or parse report: {ope}", exc_info=True)
        # Optionally include the failed output in the log/error state if helpful
        # failed_output = getattr(ope, 'llm_output', 'N/A')
        # logger.error(f"Raw LLM output causing parsing error: {failed_output[:1000]}...")
        return {"final_result": None} # Indicate failure
    except Exception as e:
        logger.error(f"[{name} - ID: {research_id}] An unexpected error occurred during final report synthesis: {e}", exc_info=True)
        return {"final_result": None}


# --- Routing Logic ---
def should_continue(state: AgentState) -> str:
    """Determines whether to continue research, finish, or handle error."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    research_id = state.get('research_id')

    # Error condition: No message
    if not last_message:
        logger.error(f"[Router - ID: {research_id}] No messages in state. Ending run.")
        return END

    # Check for explicit FINISH call in the AI message content or as a tool call name
    finish_called = False
    if isinstance(last_message, AIMessage):
        # Check if the agent called a tool named "FINISH"
        if last_message.tool_calls:
            for tc in last_message.tool_calls:
                if tc.get("name", "").upper() == "FINISH":
                    finish_called = True
                    logger.info(f"[Router - ID: {research_id}] FINISH tool call detected.")
                    break
        # Detect finish if message content mentions "FINISH" anywhere
        if not finish_called and getattr(last_message, 'content', None) and "FINISH" in last_message.content.upper():
            finish_called = True
            logger.info(f"[Router - ID: {research_id}] FINISH signal detected in message content.")

    if last_message.tool_calls and not finish_called:
        for tc in last_message.tool_calls:
            for v in tc.get('args', {}).values():
                if isinstance(v, str) and v.strip().upper() == "FINISH":
                    finish_called = True
                    logger.info(f"[Router - ID: {research_id}] FINISH signal detected in tool args.")
                    break
            if finish_called:
                break

    if finish_called:
        logger.info(f"[Router - ID: {research_id}] Routing to generate_final_report.")
        return "generate_final_report"
    # If the last message is an AIMessage with tool calls (and not FINISH), continue to tools
    elif isinstance(last_message, AIMessage) and last_message.tool_calls:
        logger.info(f"[Router - ID: {research_id}] Routing to tool_node.")
        return "tool_node"
    # If the last message was from a tool, or an AIMessage without tool calls, go back to agent
    elif isinstance(last_message, ToolMessage) or (isinstance(last_message, AIMessage) and not last_message.tool_calls):
         logger.info(f"[Router - ID: {research_id}] Routing back to agent_node.")
         return "agent_node"
    # Handle other cases or potential errors
    else:
        logger.warning(f"[Router - ID: {research_id}] Unhandled message type or state for routing: {type(last_message)}. Ending run.")
        return END


# --- Graph Construction (Renamed) ---
def create_web_research_agent_graph(config: Optional[Dict] = None):
    """Creates and compiles the LangGraph for the Web Research Agent."""
    logger.info("Creating Web Research Agent graph...")
    if config is None:
        config = {}

    # Get LLM and tools based on config
    primary_llm = get_primary_llm(config)
    if not primary_llm:
        raise ValueError("Primary LLM could not be configured.")

    # Bind tools to the LLM (common practice for OpenAI, Gemini etc.)
    tools = create_agent_tools(config)
    llm_with_tools = primary_llm.bind_tools(tools)
    tool_executor = ToolExecutor(tools)

    # Define graph nodes using partial to pass fixed args
    agent_node_partial = lambda state: agent_node(state, agent=llm_with_tools, tools=tools, name="Agent")
    tool_node_partial = lambda state: tool_node(state, tool_executor=tool_executor, name="Action")
    generate_final_report_node_partial = lambda state: generate_final_report_node(state, name="SynthesizeReport")

    # Build the graph
    workflow = StateGraph(AgentState)

    workflow.add_node("agent_node", agent_node_partial)
    workflow.add_node("tool_node", tool_node_partial)
    workflow.add_node("generate_final_report", generate_final_report_node_partial)

    # Define edges
    workflow.set_entry_point("agent_node")

    workflow.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "generate_final_report": "generate_final_report",
            "agent_node": "agent_node",
            END: END,
        },
    )

    workflow.add_edge("tool_node", "agent_node")
    workflow.add_edge("generate_final_report", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Web Research Agent graph created and compiled.")
    return app


# --- Main Execution Logic (Example Usage) ---
async def run_web_research(query: str, config: Optional[Dict] = None):
    """Runs the web research agent for a given query."""
    research_id = str(uuid.uuid4())
    logger.info(f"Starting web research for query: '{query}' (ID: {research_id})")
    if config is None:
        config = {}

    app = create_web_research_agent_graph(config)

    initial_state = AgentState(
        query=query,
        research_id=research_id,
        messages=[HumanMessage(content=query)],
        intermediate_steps=[],
        final_result=None
    )

    try:
        # LangGraph execution
        # Use stream for progress or invoke for final result
        # final_state = await app.ainvoke(initial_state, config={"recursion_limit": 15}) # Async invoke

        # Example using synchronous invoke for simplicity here
        final_state = app.invoke(initial_state, config={"recursion_limit": 40})

        logger.info(f"Research completed for ID: {research_id}")
        final_result = final_state.get('final_result')

        if isinstance(final_result, ResearchReport):
            logger.info(f"Final report generated for ID: {research_id}. Summary: {final_result.summary[:100]}...")
            return final_result
        else:
            logger.error(f"Research finished for ID: {research_id}, but no valid report was generated in the final state.")
            return ErrorResponse(error="Failed to generate report", details="The agent finished, but the final report was not found or invalid.")

    except Exception as e:
        logger.error(f"Error during research execution for ID: {research_id}: {e}", exc_info=True)
        return ErrorResponse(error="Agent execution failed", details=str(e))


# Example of how to run it (e.g., from an API endpoint)
# if __name__ == '__main__':
#     import asyncio
#     async def main():
#         # Load config from .env or elsewhere
#         # research_config = {...}
#         user_query = "What are the main challenges and opportunities for vertical farming in urban environments?"
#         result = await run_web_research(user_query)
#         if isinstance(result, ResearchReport):
#             print("\n--- Research Report ---")
#             print(f"Query: {result.query}")
#             print(f"\nSummary:\n{result.summary}")
#             for section in result.sections:
#                 print(f"\n### {section.heading}")
#                 print(section.content)
#             print("\n--- Sources ---")
#             for i, source in enumerate(result.sources):
#                 print(f"{i+1}. [{source.title}]({source.url}) - via {source.tool_used}")
#                 # print(f"   Snippet: {source.snippet}")
#             if result.potential_biases:
#                 print(f"\n--- Potential Biases/Limitations ---")
#                 print(result.potential_biases)
#         else:
#             print(f"\n--- Error ---")
#             print(result)
#
#     asyncio.run(main())

