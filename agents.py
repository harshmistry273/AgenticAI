"""
agents.py — Multi-agent orchestration system using Groq.

Architecture:
  OrchestratorAgent  — Plans tasks, routes to sub-agents, synthesizes final answer
  ResearchAgent      — Web search, Wikipedia, webpage reading
  CodeAgent          — Python execution, math/calc operations
  AnalysisAgent      — Data analysis, summaries, comparisons
  GeneralistAgent    — General Q&A, writing, reasoning
"""

import os
import json
import time
from typing import Generator, Optional
from groq import Groq
from dotenv import load_dotenv
from tools import get_tool_schemas, execute_tool

load_dotenv()


# ─────────────────────────────────────────────
#  Models (Available on Groq API key)
# ─────────────────────────────────────────────

# openai/gpt-oss-20b is fast & token-efficient
# openai/gpt-oss-120b is high-capacity for complex orchestration
MODELS = {
    "orchestrator": "openai/gpt-oss-20b",
    "research":     "openai/gpt-oss-20b",
    "code":         "openai/gpt-oss-20b",
    "analysis":     "openai/gpt-oss-20b",
    "generalist":   "openai/gpt-oss-20b",
}


# ─────────────────────────────────────────────
#  Agent definitions
# ─────────────────────────────────────────────

AGENT_CONFIGS = {
    "orchestrator": {
        "name": "🧠 Orchestrator",
        "color": "#7c3aed",
        "icon": "🧠",
        "system_prompt": """You are an expert AI orchestrator managing a team of specialized agents.
Your job is to:
1. Analyze the user's request carefully
2. Break complex tasks into clear sub-tasks
3. Delegate to the right specialist agents
4. Synthesize all results into a comprehensive, well-structured answer

Available agents:
- **ResearchAgent**: For web search, Wikipedia lookups, current events, URL reading
- **CodeAgent**: For Python execution, complex math, symbolic computation, data processing
- **AnalysisAgent**: For data analysis, comparisons, summaries, structured reports
- **GeneralistAgent**: For general knowledge, writing, creative tasks, simple Q&A

Always think step by step. If a task needs multiple agents, plan it clearly.
Format your final answers in clean markdown with headers, bullet points, and code blocks as appropriate.
Be thorough but concise.""",
        "tools": [],
    },
    "research": {
        "name": "🔍 Research Agent",
        "color": "#0891b2",
        "icon": "🔍",
        "system_prompt": """You are a specialized research agent with access to web search and Wikipedia tools.
Your expertise:
- Finding up-to-date information on any topic
- Cross-referencing multiple sources
- Extracting key facts and insights
- Providing well-cited, accurate information

Always search for information before answering factual questions.
When searching, use specific, targeted queries for best results.
Summarize findings clearly with source references.""",
        "tools": ["web_search", "wikipedia_search", "read_webpage", "get_datetime"],
    },
    "code": {
        "name": "💻 Code Agent",
        "color": "#059669",
        "icon": "💻",
        "system_prompt": """You are a specialized code and mathematics agent.
Your expertise:
- Writing and executing Python code to solve problems
- Performing complex mathematical calculations
- Data processing and transformation
- Algorithm implementation
- Unit conversions and scientific computations

Always write clean, well-commented code.
Show your work and explain the approach.
For math problems, provide step-by-step solutions.""",
        "tools": ["execute_python", "calculator", "unit_converter", "get_datetime"],
    },
    "analysis": {
        "name": "📊 Analysis Agent",
        "color": "#dc2626",
        "icon": "📊",
        "system_prompt": """You are a specialized data analysis and reasoning agent.
Your expertise:
- Analyzing complex information and data
- Creating structured comparisons and evaluations
- Identifying patterns and insights
- Writing detailed analytical reports
- Fact-checking and critical thinking

Use Python execution for quantitative analysis when needed.
Structure your analysis with clear sections and evidence-based conclusions.""",
        "tools": ["execute_python", "calculator", "web_search", "get_datetime"],
    },
    "generalist": {
        "name": "✨ Generalist Agent",
        "color": "#d97706",
        "icon": "✨",
        "system_prompt": """You are a highly capable generalist AI assistant.
Your expertise spans:
- General knowledge across all domains
- Creative writing and content generation
- Explaining complex concepts simply
- Language translation and linguistics
- Brainstorming and ideation
- Step-by-step reasoning

Provide helpful, accurate, and well-structured responses.
Format content with appropriate markdown for readability.""",
        "tools": ["get_datetime", "calculator"],
    },
}


# ─────────────────────────────────────────────
#  Core agent runner
# ─────────────────────────────────────────────

class AgentEvent:
    """Represents an event emitted during agent execution."""
    def __init__(self, agent_name: str, event_type: str, content: str, metadata: dict = None):
        self.agent_name = agent_name
        self.event_type = event_type  # "thinking", "tool_call", "tool_result", "response", "error"
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = time.time()


def run_agent(
    agent_key: str,
    messages: list[dict],
    groq_client: Groq,
    max_iterations: int = 4,
    model: Optional[str] = None,
) -> Generator[AgentEvent, None, str]:
    """
    Run an agent and yield events as it executes.
    Returns the final text response.
    """
    config = AGENT_CONFIGS[agent_key]
    model = model or MODELS.get(agent_key, "openai/gpt-oss-20b")
    agent_name = config["name"]

    # Get relevant tool schemas
    enabled_tools = config["tools"]
    all_schemas = get_tool_schemas()
    tool_schemas = [s for s in all_schemas if s["function"]["name"] in enabled_tools]

    # Build message list
    full_messages = [
        {"role": "system", "content": config["system_prompt"]},
        *messages,
    ]

    for iteration in range(max_iterations):
        try:
            # Call Groq (compact token budget to conserve quota)
            kwargs = {"model": model, "messages": full_messages, "max_tokens": 1200, "temperature": 0.5}
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                kwargs["tool_choice"] = "auto"

            response = groq_client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            # Check for tool calls
            if msg.tool_calls:
                # Add assistant message with tool calls
                full_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    yield AgentEvent(
                        agent_name=agent_name,
                        event_type="tool_call",
                        content=f"Calling **{tool_name}**",
                        metadata={"tool": tool_name, "args": tool_args},
                    )

                    # Execute tool
                    tool_result = execute_tool(tool_name, tool_args)

                    yield AgentEvent(
                        agent_name=agent_name,
                        event_type="tool_result",
                        content=tool_result,
                        metadata={"tool": tool_name, "result": tool_result},
                    )

                    # Truncate if too long to save context tokens
                    compact_result = tool_result if len(tool_result) <= 1000 else tool_result[:1000] + "... [truncated to save tokens]"
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": compact_result,
                    })

            else:
                # Final response
                final_text = msg.content or ""
                yield AgentEvent(
                    agent_name=agent_name,
                    event_type="response",
                    content=final_text,
                    metadata={"model": model, "iterations": iteration + 1},
                )
                return final_text

        except Exception as e:
            yield AgentEvent(
                agent_name=agent_name,
                event_type="error",
                content=f"Error: {str(e)}",
                metadata={"error": str(e)},
            )
            return f"Error in {agent_name}: {str(e)}"

    return "Maximum iterations reached."


# ─────────────────────────────────────────────
#  Orchestrator — multi-agent planner
# ─────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are an advanced AI orchestrator. You coordinate specialized agents to complete complex tasks.

You have access to these specialized agents:
1. **research_agent** - For web search, Wikipedia, real-time information, URLs
2. **code_agent** - For Python execution, math calculations, unit conversions
3. **analysis_agent** - For data analysis, comparisons, structured reports
4. **generalist_agent** - For general knowledge, writing, creative tasks

For each user request:
1. First, decide which agents are needed and in what order
2. Formulate specific, targeted sub-tasks for each agent
3. After receiving sub-agent results, synthesize them into a comprehensive final response

When you need to delegate, use the delegate_to_agent tool.
When you have enough information to answer fully, provide your final synthesis.

Always format your final responses beautifully with markdown:
- Use headers (##, ###) for sections
- Use bullet points and numbered lists
- Use **bold** and *italic* for emphasis  
- Use code blocks for code/data
- Be thorough but well-organized"""

DELEGATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_to_agent",
        "description": "Delegate a specific sub-task to a specialized agent",
        "parameters": {
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "enum": ["research_agent", "code_agent", "analysis_agent", "generalist_agent"],
                    "description": "Which agent to delegate to",
                },
                "task": {
                    "type": "string",
                    "description": "Clear, specific task description for the agent",
                },
                "context": {
                    "type": "string",
                    "description": "Any additional context the agent needs",
                    "default": "",
                },
            },
            "required": ["agent", "task"],
        },
    },
}

AGENT_KEY_MAP = {
    "research_agent": "research",
    "code_agent": "code",
    "analysis_agent": "analysis",
    "generalist_agent": "generalist",
}


def run_orchestrator(
    user_message: str,
    conversation_history: list[dict],
    groq_client: Groq,
    on_event=None,
    model: str = "openai/gpt-oss-20b",
) -> Generator[AgentEvent, None, None]:
    """
    Main orchestrator that plans and coordinates sub-agents.
    Yields AgentEvents for UI updates.
    """
    messages = [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
        *conversation_history,
        {"role": "user", "content": user_message},
    ]

    sub_agent_results = []
    iteration = 0
    max_iterations = 4

    while iteration < max_iterations:
        iteration += 1

        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[DELEGATION_SCHEMA],
                tool_choice="auto",
                max_tokens=1500,
                temperature=0.5,
            )

            msg = response.choices[0].message

            if msg.tool_calls:
                # Add assistant message
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    agent_key_name = args.get("agent", "generalist_agent")
                    task = args.get("task", "")
                    context = args.get("context", "")
                    agent_key = AGENT_KEY_MAP.get(agent_key_name, "generalist")
                    agent_config = AGENT_CONFIGS[agent_key]

                    # Emit delegation event
                    yield AgentEvent(
                        agent_name="🧠 Orchestrator",
                        event_type="delegation",
                        content=f"Delegating to **{agent_config['name']}**: {task}",
                        metadata={
                            "agent": agent_key_name,
                            "task": task,
                            "agent_color": agent_config["color"],
                        },
                    )

                    # Build sub-agent messages
                    sub_messages = []
                    if context:
                        sub_messages.append({"role": "user", "content": f"Context: {context}\n\nTask: {task}"})
                    else:
                        sub_messages.append({"role": "user", "content": task})

                    # Run sub-agent with token-efficient model
                    sub_final = ""
                    for event in run_agent(agent_key, sub_messages, groq_client, model=model):
                        yield event
                        if event.event_type == "response":
                            sub_final = event.content

                    sub_agent_results.append({
                        "agent": agent_config["name"],
                        "task": task,
                        "result": sub_final,
                    })

                    # Add tool result to orchestrator messages (compact to save prompt tokens)
                    compact_sub_result = sub_final if len(sub_final) <= 1000 else sub_final[:1000] + "... [summary truncated to save tokens]"
                    result_summary = f"[{agent_config['name']} Result]\nTask: {task}\n\nResult:\n{compact_sub_result}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_summary,
                    })

            else:
                # Orchestrator has final answer
                final = msg.content or ""
                yield AgentEvent(
                    agent_name="🧠 Orchestrator",
                    event_type="final_response",
                    content=final,
                    metadata={"sub_agents_used": len(sub_agent_results)},
                )
                return

        except Exception as e:
            yield AgentEvent(
                agent_name="🧠 Orchestrator",
                event_type="error",
                content=f"Orchestrator error: {str(e)}",
                metadata={"error": str(e)},
            )
            return
