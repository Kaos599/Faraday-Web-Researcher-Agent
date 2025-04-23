"""
Faraday Web Research Agent – Streamlit Interface
===============================================
A Streamlit UI for interacting with the Web Research Agent.
Run with: streamlit run app.py
"""
from __future__ import annotations

import os, json, colorsys, textwrap, asyncio # Removed requests, time; Added asyncio
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie  # Animated loaders

# Added imports for agent and schemas
from research_system.agent import run_web_research
from research_system.schemas import ResearchReport, ErrorResponse

# ────────────────────────────
# Configuration (inline) 🛠️
# ────────────────────────────
LOGO_PATH: str = os.getenv("AGENT_LOGO", "Logo.png")
PRIMARY_COLOR = "#4D96FF"  # Accent color
BG_COLOR = "#0E1117"
BG_SECONDARY = "#1B1E24"
TEXT_COLOR = "#FAFAFA"
FONT_FAMILY = "Inter, sans-serif"
LOADER_URL = "https://assets5.lottiefiles.com/private_files/lf30_editor_46utqktq.json" # Spinner animation URL
MAX_SUMMARY_WORDS = 150

# ────────────────────────────
# Helper functions
# ────────────────────────────

def hls_to_hex(hue: float, light: float = 0.5, sat: float = 0.8) -> str:
    """Convert HLS color values to a hex string."""
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def source_color(tool_name: Optional[str]) -> str:
    """Generate a consistent color based on the tool name hash."""
    if not tool_name:
        return PRIMARY_COLOR
    # Simple hash-based color generation for visual distinction
    hue = hash(tool_name) % 360 / 360.0
    return hls_to_hex(hue, light=0.6, sat=0.7)

@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> dict | None:
    """Fetch a Lottie animation and cache it."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def truncate(text: str, words: int = MAX_SUMMARY_WORDS) -> str:
    """Truncate text to a specified number of words."""
    if not text: return ""
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " …"

def render_report(report_data: Dict[str, Any]):
    """Renders the ResearchReport data."""
    if not report_data:
        st.error("No report data received.")
        return

    query = report_data.get("query", "N/A")
    summary = report_data.get("summary", "No summary provided.")
    sections = report_data.get("sections", [])
    sources = report_data.get("sources", [])
    biases = report_data.get("potential_biases")

    st.subheader(f"Research Report for: \"{query}\"")

    # --- Display Summary ---
    st.markdown("### Executive Summary")
    summary_truncated = truncate(summary)
    st.markdown(f"<div class='summary-box'>{summary_truncated}</div>", unsafe_allow_html=True)
    if len(summary.split()) > MAX_SUMMARY_WORDS:
        with st.expander("Read full summary"):
            st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

    # --- Display Sections ---
    if sections:
        st.markdown("### Detailed Findings")
        for section in sections:
            heading = section.get('heading', 'Section')
            content = section.get('content', 'No content.')
            with st.expander(heading, expanded=False):
                 st.markdown(content, unsafe_allow_html=True)
    else:
        st.info("No detailed sections were generated in the report.")

    # --- Display Potential Biases/Limitations ---
    if biases:
        st.markdown("### Potential Biases & Limitations")
        st.warning(biases)

    # --- Display Sources ---
    if sources:
        st.markdown("### Sources Consulted")
        # Create tool badges at the top
        tools_used = sorted({src.get("tool_used", "Unknown") for src in sources if src.get("tool_used")})
        if tools_used:
            with st.container():
                tool_cols = st.columns(min(len(tools_used), 4))
                for i, tool in enumerate(tools_used):
                    col_index = i % 4
                    tool_badge_color = source_color(tool)
                    tool_cols[col_index].markdown(
                        f"""<div style='background:{tool_badge_color}33;padding:8px 12px;
                        border-radius:6px;font-weight:600;text-align:center;font-size:0.9em;
                        margin-bottom:10px;border:1px solid {tool_badge_color}55;
                        box-shadow:0 2px 4px rgba(0,0,0,0.1);'>{tool}</div>""",
                        unsafe_allow_html=True,
                    )
            st.markdown("""<div style='height:15px'></div>""", unsafe_allow_html=True)

        # Display each source
        for i, src in enumerate(sources):
            # Generate color based on tool used
            tool_name = src.get("tool_used", "Web Source")
            c = source_color(tool_name)
            title = src.get("title") or src.get("url", "No title")
            snippet = src.get("snippet", "No preview available")
            url = src.get("url")

            st.markdown(
                f"""
                <div style='border-left:6px solid {c};padding:15px;margin:15px 0;
                    border-radius:8px;background:{BG_SECONDARY};position:relative;
                    box-shadow:0 4px 6px rgba(0, 0, 0, 0.2);'>
                    <span style='position:absolute;top:10px;right:10px;background:{c};color:white;
                        padding:4px 10px;border-radius:4px;font-size:12px;font-weight:bold;'>{tool_name}</span>
                    <a href='{url}' target='_blank'
                        style='color:{PRIMARY_COLOR};font-weight:600;
                        text-decoration:none;font-size:16px;display:block;margin-top:5px;margin-bottom:10px;'>{i+1}. {title}</a>
                    <span style='color:#CCCCCC;font-size:14px;'>{snippet}</span>
                    <div style='clear:both;'></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No sources were listed in the final report.")


# ────────────────────────────
# Global page settings
# ────────────────────────────
st.set_page_config(
    page_title="Faraday Web Research Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={
        'About': "Faraday Web Research Agent - An AI assistant to research topics online."
    }
)

# ────────────────────────────
# Custom CSS styling
# ────────────────────────────
st.markdown(
    f"""
    <style>
    /* Root variables for theming */
    :root {{
        --primary-color: {PRIMARY_COLOR};
        --text-color: {TEXT_COLOR};
        --background-color: {BG_COLOR};
        --secondary-background-color: {BG_SECONDARY};
        --font: {FONT_FAMILY};
    }}

    /* Base styling */
    html, body, [class*="st"] {{
        font-family: var(--font);
    }}

    .stApp {{
        background-color: var(--background-color);
        color: var(--text-color);
    }}

    a {{
        color: var(--primary-color);
    }}

    /* Logo size adjustment */
    .logo-container img {{
        width: auto !important;
        height: 80px !important;
    }}

    /* Search bar styling */
    .stTextInput > div > div > input {{
        text-align: center;
        font-size: 1.25em;
        background-color: {BG_SECONDARY};
        color: white;
        border-radius: 25px;
        border: 1px solid {PRIMARY_COLOR};
        padding: 12px 20px;
    }}

    /* Main container */
    .main-container {{
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }}

    /* Logo container */
    .logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
        margin-top: 30px;
    }}

    /* Summary box */
    .summary-box {{
        background-color: {BG_SECONDARY};
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #333;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        line-height: 1.6;
    }}

    /* Loading steps */
    .loader-step {{
        display: flex;
        align-items: center;
        margin: 10px 0;
        padding: 12px 15px;
    }}

    .loader-step-active {{
        border-left: 3px solid {PRIMARY_COLOR};
        box-shadow: 0 0 8px {PRIMARY_COLOR}40;
    }}

    .loader-step-complete {{
        border-left: 3px solid #00CC66;
        box-shadow: 0 0 8px #00CC6640;
    }}

    .loader-icon {{
        margin-right: 15px;
        font-size: 18px;
    }}

    /* Additional tweaks */
    h1, h2, h3 {{
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }}

    .stApp > header {{
        background-color: transparent;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────
# Main app layout
# ────────────────────────────
# Header with logo
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
if os.path.exists(LOGO_PATH):
    try:
        logo_image = Image.open(LOGO_PATH)
        st.image(logo_image, width=900, output_format="PNG", use_container_width=False, caption="")
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        st.markdown("<h1 style='text-align:center;margin-bottom:0'>Faraday Web Research Agent</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align:center;margin-bottom:0'>Faraday Web Research Agent</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#888;margin-top:4px'>Your AI Research Assistant</h4>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Research query input bar
query_input = st.text_input(
    "Research Query",
    placeholder="Enter your research query...",
    label_visibility="collapsed"
)

# Initialize session state for tracking progress
if 'progress_state' not in st.session_state:
    st.session_state.progress_state = {
        'status': 'idle', # idle, running, completed, error
        'error_message': None,
        'report_data': None, # Renamed from 'api_data'
        'current_query': None # Store the query associated with the current state
    }

# Results container
results_container = st.container()

if query_input:
    with results_container:
        current_status = st.session_state.progress_state['status']
        stored_query = st.session_state.progress_state.get('current_query')

        # --- Check if the query has changed --- NEW LOGIC
        if query_input != stored_query:
            # User entered a new query, reset state and start running
            st.session_state.progress_state['status'] = 'running'
            st.session_state.progress_state['error_message'] = None
            st.session_state.progress_state['report_data'] = None
            st.session_state.progress_state['current_query'] = query_input
            st.rerun()
        # --- END NEW LOGIC ---

        # Proceed with existing logic only if the query HAS NOT changed
        # --- Start the process if status is idle --- (This case might become less frequent)
        elif current_status == 'idle':
            # This could happen on the very first run after initial load
            st.session_state.progress_state['status'] = 'running'
            st.session_state.progress_state['error_message'] = None
            st.session_state.progress_state['report_data'] = None
            st.session_state.progress_state['current_query'] = query_input # Store query when starting
            st.rerun() # Trigger rerun to show spinner and start processing

        # --- Run the research directly if status is running ---
        elif current_status == 'running':
            # Show animated steps while running
            st.markdown('<div style="margin: 30px 0;">', unsafe_allow_html=True)
            step1_class = "loader-step loader-step-active" # Now active
            st.markdown(f'<div class="{step1_class}"><span class="loader-icon">🧠</span> Thinking & Researching...</div>', unsafe_allow_html=True)
            step2_class = "loader-step" # Not active yet
            st.markdown(f'<div class="{step2_class}"><span class="loader-icon">📄</span> Gathering & Synthesizing Information...</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Use spinner while the agent runs
            with st.spinner("Performing web research... This may take a few moments."):
                try:
                    # Directly call the agent function
                    research_result = asyncio.run(run_web_research(query=query_input))

                    # Check the result type and update state
                    if isinstance(research_result, ResearchReport):
                        st.session_state.progress_state['status'] = 'completed'
                        st.session_state.progress_state['report_data'] = research_result.dict() # Store report as dict
                    elif isinstance(research_result, ErrorResponse):
                        st.session_state.progress_state['status'] = 'error'
                        st.session_state.progress_state['error_message'] = f"{research_result.error}: {research_result.details}"
                    else:
                        # Handle unexpected return type
                        st.session_state.progress_state['status'] = 'error'
                        st.session_state.progress_state['error_message'] = f"Agent returned an unexpected result type: {type(research_result)}"

                    st.rerun() # Rerun to display results or error

                except ImportError as ie:
                     st.session_state.progress_state['status'] = 'error'
                     st.session_state.progress_state['error_message'] = f"Import Error: {ie}. Ensure agent components are installed and accessible."
                     st.rerun()
                except Exception as e:
                    st.session_state.progress_state['status'] = 'error'
                    st.session_state.progress_state['error_message'] = f"An unexpected error occurred during research: {e}"
                    st.rerun()

        # --- Display results if process is complete ---
        elif current_status == 'completed':
            report_data = st.session_state.progress_state.get('report_data') # Use renamed key
            if report_data:
                 render_report(report_data)

                 # Reset progress state if user wants to search again - REMOVED BUTTON
                 # if st.button("New Research Query", use_container_width=True, type="primary"):
                 #     st.session_state.progress_state = {
                 #         'status': 'idle',
                 #         'error_message': None,
                 #         'report_data': None,
                 #         'current_query': None
                 #     }
                 #     st.rerun()
                 st.info("Enter a new query above to start another research task.") # Inform user

            else:
                 st.error("Completed status reached but no report data found.")
                 # Reset to allow trying again
                 st.session_state.progress_state['status'] = 'idle'
                 st.session_state.progress_state['current_query'] = None # Clear query too
                 st.rerun()

        # --- Display error if status is error ---
        elif current_status == 'error':
            st.error(f"Research failed: {st.session_state.progress_state.get('error_message', 'Unknown error')}")
            # Allow user to try again - REMOVED BUTTON
            # if st.button("Try New Research", use_container_width=True, type="primary"):
            #     st.session_state.progress_state = {
            #         'status': 'idle',
            #         'error_message': None,
            #         'report_data': None,
            #         'current_query': None
            #     }
            #     st.rerun()
            st.info("Enter a new query above to try again or start a new research task.") # Inform user

else:
    # Show prompt if no query is entered
    with results_container:
        st.markdown("""
        <div style="text-align: center; margin-top: 50px; color: #AAAAAA;">
            <h3>Enter a research query above to start</h3>
            <p>Example: "What are the pros and cons of universal basic income?"</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("</div>", unsafe_allow_html=True)  # Close main container
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #888; font-size: 0.8em;">
    <p>Faraday Web Research Agent • Powered by AI</p>
</div>
""", unsafe_allow_html=True)
