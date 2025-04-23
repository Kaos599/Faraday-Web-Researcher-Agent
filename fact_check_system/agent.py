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

    # Format the evidence from intermediate steps for the synthesis prompt
    formatted_evidence = ""
    collected_sources_for_report: List[Source] = []
    unique_source_urls = set()

    logger.info(f"[{name} - ID: {research_id}] Formatting evidence from {len(intermediate_steps)} intermediate steps.")
    for step_index, (tool_invocation, observation) in enumerate(intermediate_steps):
        step_number = step_index + 1
        tool_name = tool_invocation.tool
        tool_input = tool_invocation.tool_input
        formatted_evidence += f"\n--- Step {step_number}: Tool Used: {tool_name} ---\n"
        formatted_evidence += f"Input: {str(tool_input)[:200]}{'...' if len(str(tool_input)) > 200 else ''}\n"
        formatted_evidence += f"Observation:\n{str(observation)[:1000]}{'...' if len(str(observation)) > 1000 else ''}\n"

        # Basic source extraction attempt from observation (can be refined)
        # This is a simple heuristic; ideally, tools return structured source info
        try:
            obs_data = json.loads(observation) if isinstance(observation, str) else observation
            if isinstance(obs_data, dict):
                results = obs_data.get('results') or obs_data.get('articles') or obs_data.get('search')
                if isinstance(results, list):
                    for item in results:
                        if isinstance(item, dict):
                            url = item.get('url') or item.get('link') or item.get('concepturi')
                            title = item.get('title') or item.get('label')
                            snippet = item.get('snippet') or item.get('content') or item.get('description')
                            if url and url not in unique_source_urls:
                                collected_sources_for_report.append(
                                    Source(url=url, title=title, snippet=snippet, tool_used=tool_name)
                                )
                                unique_source_urls.add(url)
        except Exception as e:
            logger.warning(f"[{name} - ID: {research_id}] Error parsing sources from step {step_number} observation: {e}")
            pass # Continue even if source parsing fails for a step

    logger.info(f"[{name} - ID: {research_id}] Total unique sources extracted for report: {len(collected_sources_for_report)}")
    formatted_evidence += "\n-- End of Evidence --\n"

    # Prepare the prompt for the synthesis LLM
    synthesis_prompt = REPORT_SYNTHESIS_TEMPLATE.format(
        query=original_query,
        formatted_evidence=formatted_evidence
        # format_instructions is already partialled in prompts.py
    )

    logger.info(f"[{name} - ID: {research_id}] Invoking synthesis LLM...")
    try:
        synthesis_response = synthesis_model.invoke(synthesis_prompt)
        logger.info(f"[{name} - ID: {research_id}] Synthesis LLM response received. Type: {type(synthesis_response)}")
        # logger.debug(f"[{name}] Synthesis LLM raw response content: {synthesis_response.content[:500]}...")

        # Parse the response using the PydanticOutputParser
        if hasattr(synthesis_response, 'content'):
            final_report: ResearchReport = report_parser.parse(synthesis_response.content)
            # Add the programmatically extracted sources to the report
            # This overrides any sources the LLM might have hallucinated in the list
            # but keeps the LLM's summary and sections.
            final_report.sources = collected_sources_for_report
            logger.info(f"[{name} - ID: {research_id}] Successfully parsed research report.")
            return {"final_result": final_report}
        else:
            logger.error(f"[{name} - ID: {research_id}] Synthesis LLM response has no 'content' attribute: {synthesis_response}")
            return {"final_result": None}

    except Exception as e:
        logger.error(f"[{name} - ID: {research_id}] Failed to invoke synthesis LLM or parse report: {e}", exc_info=True)
        # Optionally, try to save the raw response text in the report if parsing fails
        # error_report = ResearchReport(query=original_query, summary="Error during synthesis.", sections=[ResearchReportSection(heading="Error", content=f"Failed to generate report: {e}")], sources=[])
        # return {"final_result": error_report}
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

