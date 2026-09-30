# 🤖 Advanced Agentic AI System

A production-grade **multi-agent AI system** built with Streamlit and powered by **Groq** (Llama 3.3 70B).

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multi-Agent Orchestration** | Orchestrator routes tasks to 4 specialized agents |
| 🔍 **Web Search** | Real-time DuckDuckGo search integration |
| 🌐 **Webpage Reader** | Fetch and extract content from any URL |
| 📚 **Wikipedia** | Instant encyclopedic knowledge lookup |
| 💻 **Code Executor** | Safe Python sandbox for computation |
| 🧮 **Symbolic Math** | SymPy-powered calculus, algebra, equations |
| 📏 **Unit Converter** | Length, weight, temperature, data, speed |
| 🕐 **Date/Time** | Timezone-aware datetime operations |

## 🤖 Agent Team

```
OrchestratorAgent (Llama 3.3 70B)
├── 🔍 ResearchAgent  → web_search, wikipedia, read_webpage
├── 💻 CodeAgent      → execute_python, calculator, unit_converter
├── 📊 AnalysisAgent  → execute_python, calculator, web_search
└── ✨ GeneralistAgent → calculator, get_datetime
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open browser
Navigate to `http://localhost:8501`

The Groq API key is pre-configured. You can change it in the sidebar settings.

## 📁 Project Structure

```
agentic-ai-streamlit/
├── app.py              ← Streamlit UI (main entry point)
├── agents.py           ← Multi-agent orchestration system
├── tools.py            ← Tool registry (7 tools)
├── requirements.txt    ← Python dependencies
├── .streamlit/
│   └── config.toml     ← Dark theme config
└── README.md           ← This file
```

## 💡 Example Prompts

- *"What are the latest AI breakthroughs in 2024?"* → Research Agent searches the web
- *"Solve the integral of x² × sin(x)"* → Code Agent uses SymPy
- *"Compare Python vs JavaScript for web dev"* → Analysis Agent provides structured analysis
- *"What time is it in Tokyo, NYC, and London?"* → Datetime tool
- *"Write Python to compute prime numbers up to 1000"* → Code Agent executes Python

## 🛠️ Extending

**Add a new tool** in `tools.py`:
```python
@register_tool(
    name="my_tool",
    description="What this tool does",
    parameters={
        "type": "object",
        "properties": {"param": {"type": "string"}},
        "required": ["param"],
    },
)
def my_tool(param: str) -> str:
    return json.dumps({"result": f"Processed: {param}"})
```

**Add a new agent** in `agents.py` — add an entry to `AGENT_CONFIGS` with its system prompt and enabled tools.

## 🔧 Configuration

| Setting | Value |
|---------|-------|
| Orchestrator Model | `llama-3.3-70b-versatile` |
| Sub-agent Models | `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` |
| Max Tool Iterations | 8 per agent |
| Max Orchestration Steps | 10 |

## 📦 Dependencies

- `streamlit` — Web UI framework
- `groq` — Groq API client (Llama 3.3)
- `duckduckgo-search` — Free web search
- `beautifulsoup4` — HTML parsing
- `sympy` — Symbolic mathematics
- `pytz` — Timezone support
- `requests` — HTTP client
- `plotly` — Data visualization (ready to use)
- `pandas` — Data manipulation (ready to use)
