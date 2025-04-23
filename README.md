# Faraday Web Research Agent 🕵️‍♀️

![Faraday Logo](Logo.png)

[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Orchestration](https://img.shields.io/badge/Orchestration-LangChain%20/%20LangGraph-purple.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Overview

Faraday is a comprehensive web research agent designed to investigate queries by **autonomously gathering and analyzing information** from multiple online sources. It uses a sophisticated agent powered by LLMs and LangGraph to dynamically select tools, conduct research, and synthesize findings, ultimately delivering a structured research report.

## 🚀 Features

- **🤖 Agentic Workflow**: Employs an AI agent orchestrated with LangGraph to manage the entire research process.
- **🛠️ Dynamic Tool Selection**: The agent intelligently chooses the best tools (Search Engines, Web Scrapers, APIs) based on the query and intermediate findings.
- **🔍 Multi-source Evidence Collection**: Gathers information from diverse sources using tools like Tavily, Google Search (via Gemini), DuckDuckGo, Wikidata, NewsAPI, and Firecrawl.
- **🧩 Query Decomposition**: Can break down complex queries into simpler, searchable sub-questions using LLMs.
- **📝 Structured Reporting**: Synthesizes findings into a well-organized report with a summary, detailed sections, and source attribution.
- **🔗 Source Attribution**: Transparently lists all sources consulted and the tools used to access them.
- **🖥️ Modern Dark Mode Interface**: Clean, user-friendly Streamlit interface for interaction and result presentation.

## 🏗️ System Architecture

Faraday leverages an **agentic architecture**, orchestrated using LangGraph. Instead of a fixed pipeline, a central **Web Research Agent** dynamically plans and executes tasks using a suite of available tools:

```mermaid
architecture-beta
    node User
    node Streamlit_UI[Streamlit Frontend]
    node Backend_API[FastAPI Backend API]
    node LangGraph_Agent[LangGraph Agent]
    node Primary_LLM[Primary LLM] : Decision Making
    node Parser_LLM[Parser LLM] : Gemini Output Parsing
    group Tools
        node Tavily[Tavily Search]
        node DuckDuckGo[DuckDuckGo Search]
        node GeminiSearch[Gemini Search Tool]
        node Firecrawl[Firecrawl Scrape Tool]
        node NewsAPI[News API Tool]
        node Wikidata[Wikidata Search Tool]
        node QueryDecomp[Query Decomposition Tool]
        node FINISH[FINISH Signal]
    end
    group External_Sources
        node Web[Websites]
        node DataSources[APIs/Databases]
    end
    node Research_Report[Research Report (Schema)]

    User --|> Streamlit_UI : Inputs Query
    Streamlit_UI --|> Backend_API : API Request
    Backend_API --|> LangGraph_Agent : Invokes Agent
    LangGraph_Agent --|> Primary_LLM : Reasoning & Tool Selection
    LangGraph_Agent --|> Parser_LLM : Parses Specific Outputs
    LangGraph_Agent --|> Tools : Tool Invocation
    Tools --|> External_Sources : Data Retrieval
    External_Sources --|> Tools : Returns Data
    Tools --|> LangGraph_Agent : Observations
    LangGraph_Agent --> Research_Report : Synthesizes Report
    LangGraph_Agent --> Backend_API : Returns Report/Status
    Backend_API --> Streamlit_UI : Sends Report/Status
    Streamlit_UI --> User : Displays Report

```

*This diagram represents a high-level overview of the system components and their interactions, driven by the agent's dynamic workflow.*

## ⚙️ How the System Works (Agentic Flow)

The web research process is driven by the agent's autonomous reasoning within the LangGraph framework:

1.  **User Input**: A user submits a research query via the Streamlit UI.
2.  **API Request**: The frontend sends the query to the backend API, initiating the agent.
3.  **Agent Execution**: The LangGraph agent manages a state and iteratively performs steps:
    *   It analyzes the query and decides the next best action (e.g., decompose query, search the web, scrape a page).
    *   It invokes the appropriate tool with specific inputs.
    *   It processes the tool's observation (output) and updates its state.
    *   This cycle continues (Agent -> Tool -> Agent) until the agent determines it has gathered sufficient information.
4.  **Final Synthesis**: Once the agent decides to finish, it synthesizes all gathered information and structured observations into a `ResearchReport` object.
5.  **Presentation**: The final report is returned via the API and presented to the user in the Streamlit interface.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required API keys stored securely (e.g., in a `.env` file) for the tools you intend to use (e.g., OpenAI, Google AI, Tavily, NewsAPI, Firecrawl).

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up environment variables by creating a `.env` file in the root directory with your API keys.

### Running the Application

1.  Start the backend API server:
    ```bash
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    ```
    The API will typically be available at `http://127.0.0.1:8000`.

2.  Start the Streamlit frontend in a separate terminal:
    ```bash
    streamlit run app.py
    ```
    The app will usually be available at `http://localhost:8501`.

## 🤔 Example Queries to Try

Challenge the agent with various research queries:

- "What are the latest advancements in renewable energy technology?"
- "Explain the concept of large language models and their applications."
- "Compare and contrast the economic impacts of Brexit on the UK and the EU."
- "Provide a brief history of the internet."

## 🔌 API Usage

The system provides a REST API endpoint to trigger the web research agent:

### Research Query

```
POST /research
Content-Type: application/json

{
  "query": "Your research query here",
  "language": "en" // Optional, defaults might apply
}
```

**Example Success Response (Status 202 Accepted, Polling Required):**

```json
{
  "task_id": "unique-task-identifier",
  "status": "processing"
}
```

**Polling for Results:**

```
GET /results/{task_id}
```

**Example Completed Response:**

```json
{
  "task_id": "unique-task-identifier",
  "status": "completed",
  "result": { // ResearchReport object
    "query": "Your research query here",
    "summary": "Executive summary generated by the agent...",
    "sections": [
      { "heading": "Section Heading", "content": "Detailed content..." }
      // ... more sections
    ],
    "sources": [
      { "url": "https://example.com/source1", "title": "Source Title", "snippet": "...", "tool_used": "..." }
      // ... other sources
    ],
    "potential_biases": "Any identified biases or limitations...",
    "report_generated_at": "timestamp"
  }
}
```

**Example Error Response:**

```json
{
  "task_id": "unique-task-identifier",
  "status": "error",
  "error": { // ErrorResponse object
    "error": "Error type or summary",
    "details": "More detailed error message..."
  }
}
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to add new tools or features, please feel free to:

1.  Open an issue to discuss the change.
2.  Fork the repository.
3.  Create a new branch (`git checkout -b feature/YourFeature`).
4.  Make your changes.
5.  Commit your changes (`git commit -m 'Add some feature'`).
6.  Push to the branch (`git push origin feature/YourFeature`).
7.  Open a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- Built with ❤️ using [Python](https://www.python.org/), [Streamlit](https://streamlit.io/), and [LangChain/LangGraph](https://www.langchain.com/).
- Leverages powerful APIs and tools from [Tavily AI](https://tavily.com/), [Google AI (Gemini)](https://ai.google.dev/), [DuckDuckGo](https://duckduckgo.com/), [Firecrawl](https://firecrawl.dev/), [NewsAPI](https://newsapi.org/), [Wikidata](https://www.wikidata.org/), and others.
- Inspired by the need for effective and automated information gathering. 