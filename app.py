"""
app.py — Streamlit UI for the Advanced Agentic AI System.

Features:
  - Real-time streaming agent activity
  - Multi-agent orchestration visualization
  - Conversation history with context
  - Tool usage display
  - Settings panel
  - Example prompts
"""

import streamlit as st
import os
import json
import time
from dotenv import load_dotenv
from groq import Groq
from agents import run_orchestrator, AGENT_CONFIGS, AgentEvent

load_dotenv()

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🤖 Agentic AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ─────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }

/* ── Chat messages ─────────────────────── */
.chat-msg {
    padding: 14px 18px;
    border-radius: 14px;
    margin: 8px 0;
    line-height: 1.6;
    animation: fadeIn 0.3s ease;
}
.chat-user {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white;
    margin-left: 15%;
    border-bottom-right-radius: 4px;
}
.chat-assistant {
    background: #1e293b;
    color: #e2e8f0;
    margin-right: 15%;
    border: 1px solid #334155;
    border-bottom-left-radius: 4px;
}
.chat-meta {
    font-size: 0.72rem;
    opacity: 0.6;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Agent event cards ─────────────────── */
.agent-card {
    padding: 10px 14px;
    border-radius: 10px;
    margin: 6px 0;
    border-left: 4px solid;
    font-size: 0.88rem;
    animation: slideIn 0.2s ease;
}
.agent-orchestrator  { border-color: #7c3aed; background: #1a1040; }
.agent-research      { border-color: #0891b2; background: #062b38; }
.agent-code          { border-color: #059669; background: #052e22; }
.agent-analysis      { border-color: #dc2626; background: #300a0a; }
.agent-generalist    { border-color: #d97706; background: #2a1f05; }

.event-tool-call   { border-color: #f59e0b !important; background: #1f1500 !important; }
.event-tool-result { border-color: #10b981 !important; background: #052e1a !important; }
.event-delegation  { border-color: #8b5cf6 !important; background: #1a0d40 !important; }
.event-error       { border-color: #ef4444 !important; background: #300a0a !important; }

/* ── Tool result code ──────────────────── */
.tool-result-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    max-height: 200px;
    overflow-y: auto;
    color: #94a3b8;
    margin-top: 6px;
    white-space: pre-wrap;
}

/* ── Stats badges ─────────────────────── */
.stat-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Sidebar ───────────────────────────── */
[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ── Input box ─────────────────────────── */
.stTextInput input, .stTextArea textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── Buttons ───────────────────────────── */
.stButton button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s;
}
.stButton button:hover { transform: translateY(-1px); }

/* ── Animations ────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.5; }
}
.thinking-dot {
    animation: pulse 1.2s infinite;
    display: inline-block;
    font-size: 1.5rem;
}

/* ── Agent status indicator ────────────── */
.agent-status {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    background: #1e293b;
    margin: 4px 0;
    font-size: 0.85rem;
    border: 1px solid #334155;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
}

/* ── Example prompts ───────────────────── */
.example-chip {
    display: inline-block;
    padding: 6px 14px;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    font-size: 0.8rem;
    color: #94a3b8;
    cursor: pointer;
    margin: 3px;
    transition: all 0.2s;
}
.example-chip:hover {
    background: #334155;
    color: #e2e8f0;
    border-color: #7c3aed;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Session State Init
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "messages": [],          # [{role, content, events}]
        "is_processing": False,
        "groq_api_key": st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "")),
        "selected_model": "openai/gpt-oss-20b",
        "show_tool_results": True,
        "show_thinking": True,
        "total_tool_calls": 0,
        "total_agents_used": 0,
        "total_messages": 0,
        "input_key": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 Agentic AI System")
    st.markdown("*Powered by Groq Cloud*")
    st.divider()

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 Messages", st.session_state.total_messages)
        st.metric("🔧 Tool Calls", st.session_state.total_tool_calls)
    with col2:
        st.metric("🤖 Agents Used", st.session_state.total_agents_used)
        st.metric("💾 History", len(st.session_state.messages))

    st.divider()

    # Agent Legend
    st.markdown("### 🤖 Agent Team")
    for key, config in AGENT_CONFIGS.items():
        st.markdown(
            f'<div class="agent-status">'
            f'<div class="status-dot" style="background:{config["color"]}"></div>'
            f'<span style="font-size:0.85rem">{config["name"]}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Settings
    st.markdown("### ⚙️ Settings")

    available_models = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]
    st.session_state.selected_model = st.selectbox(
        "Groq Model",
        options=available_models,
        index=0,
        help="openai/gpt-oss-20b is ultra-fast and conserves token budget.",
    )

    st.session_state.show_tool_results = st.toggle(
        "Show Tool Results", value=st.session_state.show_tool_results
    )
    st.session_state.show_thinking = st.toggle(
        "Show Agent Activity", value=st.session_state.show_thinking
    )
    groq_key = st.text_input(
        "Groq API Key",
        value=st.session_state.groq_api_key,
        type="password",
        help="Your Groq API key",
    )
    if groq_key:
        st.session_state.groq_api_key = groq_key

    st.divider()

    # Clear
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tool_calls = 0
        st.session_state.total_agents_used = 0
        st.session_state.total_messages = 0
        st.rerun()

    st.divider()
    st.markdown(
        '<div style="font-size:0.72rem;color:#64748b;text-align:center">'
        'Built with Streamlit + Groq<br>'
        'Llama 3.3 70B · Multi-Agent · RAG'
        '</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  Main Layout
# ─────────────────────────────────────────────

st.markdown(
    '<h1 style="text-align:center;font-size:2rem;margin-bottom:0">'
    '🤖 Advanced Agentic AI System</h1>'
    '<p style="text-align:center;color:#64748b;margin-top:4px;font-size:0.9rem">'
    'Multi-agent orchestration · Real-time tool use · Groq-powered intelligence'
    '</p>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Example Prompts
# ─────────────────────────────────────────────

EXAMPLES = [
    "🔭 What are the latest AI breakthroughs in 2024?",
    "🧮 Solve the integral of x² * sin(x) step by step",
    "📊 Analyze the pros and cons of Python vs JavaScript",
    "🌐 Search and summarize: quantum computing progress",
    "💻 Write Python to generate Fibonacci numbers & plot them",
    "🕐 What time is it in Tokyo, New York, and London right now?",
    "🔬 Explain how CRISPR gene editing works",
    "📈 Convert 10,000 USD to multiple currencies (context only)",
    "🎯 Plan a 7-day learning roadmap for machine learning",
    "🔢 Calculate the compound interest on $5000 at 7% for 10 years",
]

if not st.session_state.messages:
    st.markdown("---")
    st.markdown("#### 💡 Try these example prompts:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(example, key=f"ex_{i}", use_container_width=True):
                st.session_state["prefill"] = example

# ─────────────────────────────────────────────
#  Render existing messages
# ─────────────────────────────────────────────

def render_event(event: dict):
    """Render a single agent event."""
    etype = event.get("event_type", "")
    content = event.get("content", "")
    agent_name = event.get("agent_name", "")
    metadata = event.get("metadata", {})

    if etype == "delegation":
        st.markdown(
            f'<div class="agent-card event-delegation">'
            f'🎯 <strong>Orchestrator</strong> → {content}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif etype == "tool_call":
        tool = metadata.get("tool", "")
        args = metadata.get("args", {})
        args_str = json.dumps(args, indent=2) if args else "{}"
        st.markdown(
            f'<div class="agent-card event-tool-call">'
            f'🔧 <strong>{agent_name}</strong> — {content}'
            f'<div class="tool-result-box">{args_str}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif etype == "tool_result" and st.session_state.show_tool_results:
        try:
            result_data = json.loads(content)
            display = json.dumps(result_data, indent=2)
        except Exception:
            display = content
        # Truncate for display
        if len(display) > 800:
            display = display[:800] + "\n... (truncated)"
        st.markdown(
            f'<div class="agent-card event-tool-result">'
            f'✅ <strong>Tool Result</strong>'
            f'<div class="tool-result-box">{display}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif etype == "response":
        model = metadata.get("model", "")
        iters = metadata.get("iterations", 1)
        st.markdown(
            f'<div class="agent-card agent-{agent_name.split()[1].lower() if len(agent_name.split()) > 1 else "generalist"}">'
            f'💬 <strong>{agent_name}</strong> responded '
            f'<span style="opacity:0.6;font-size:0.8rem">({iters} step{"s" if iters > 1 else ""})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif etype == "error":
        st.markdown(
            f'<div class="agent-card event-error">'
            f'❌ <strong>Error in {agent_name}</strong>: {content}'
            f'</div>',
            unsafe_allow_html=True,
        )


chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        events = msg.get("events", [])

        if role == "user":
            st.markdown(
                f'<div class="chat-msg chat-user">'
                f'<div class="chat-meta">👤 You</div>'
                f'{content}'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif role == "assistant":
            # Show agent activity
            if events and st.session_state.show_thinking:
                with st.expander("🔍 Agent Activity Log", expanded=False):
                    for event in events:
                        render_event(event)

            st.markdown(
                f'<div class="chat-msg chat-assistant">'
                f'<div class="chat-meta">🤖 Agentic AI</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(content)

# ─────────────────────────────────────────────
#  Input Area
# ─────────────────────────────────────────────

st.markdown("---")
prefill_val = st.session_state.pop("prefill", "")

with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_area(
            "Message",
            value=prefill_val,
            placeholder="Ask me anything... I'll coordinate multiple AI agents to answer! 🚀",
            label_visibility="collapsed",
            height=80,
            key=f"input_{st.session_state.input_key}",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Send 🚀", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  Process user input
# ─────────────────────────────────────────────

def process_message(user_text: str):
    """Run the orchestrator and stream results to UI."""
    if not st.session_state.groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.total_messages += 1

    # Build conversation history (last 10 turns)
    history = []
    for m in st.session_state.messages[-20:]:
        if m["role"] in ("user", "assistant"):
            history.append({"role": m["role"], "content": m["content"]})

    # Groq client
    client = Groq(api_key=st.session_state.groq_api_key)

    # Live activity area
    activity_placeholder = st.empty()
    events_log = []
    final_response = ""
    tool_call_count = 0
    agents_used = set()

    activity_placeholder.markdown(
        '<div style="padding:16px;background:#1e293b;border-radius:12px;border:1px solid #334155">'
        '<span class="thinking-dot">⚡</span> '
        '<strong>Orchestrating agents...</strong>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Stream events
    event_display = []
    for event in run_orchestrator(user_text, history[:-1], client, model=st.session_state.selected_model):
        events_log.append({
            "agent_name": event.agent_name,
            "event_type": event.event_type,
            "content": event.content,
            "metadata": event.metadata,
        })

        if event.event_type == "tool_call":
            tool_call_count += 1
        if event.event_type in ("response", "final_response", "delegation"):
            agents_used.add(event.agent_name)
        if event.event_type == "final_response":
            final_response = event.content

        # Update live display
        if st.session_state.show_thinking:
            event_display.append(event)
            activity_html = '<div style="padding:16px;background:#0f172a;border-radius:12px;border:1px solid #1e293b;max-height:300px;overflow-y:auto">'
            activity_html += '<div style="margin-bottom:10px;font-weight:600;color:#94a3b8">🔄 Agent Activity (Live)</div>'

            for ev in event_display[-8:]:  # Show last 8 events
                etype = ev.event_type
                aname = ev.agent_name
                content_preview = ev.content[:120] + "..." if len(ev.content) > 120 else ev.content

                if etype == "delegation":
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #8b5cf6;background:#1a0d40;border-radius:4px;font-size:0.82rem;color:#c4b5fd">🎯 {content_preview}</div>'
                elif etype == "tool_call":
                    tool = ev.metadata.get("tool", "")
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #f59e0b;background:#1f1500;border-radius:4px;font-size:0.82rem;color:#fcd34d">🔧 {aname} → {tool}</div>'
                elif etype == "tool_result":
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #10b981;background:#052e1a;border-radius:4px;font-size:0.82rem;color:#6ee7b7">✅ Tool result received</div>'
                elif etype == "response":
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #3b82f6;background:#0c1a40;border-radius:4px;font-size:0.82rem;color:#93c5fd">💬 {aname} completed</div>'
                elif etype == "final_response":
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #22c55e;background:#052e1a;border-radius:4px;font-size:0.82rem;color:#86efac">✨ Final response ready!</div>'
                elif etype == "error":
                    activity_html += f'<div style="padding:6px 10px;margin:3px 0;border-left:3px solid #ef4444;background:#300a0a;border-radius:4px;font-size:0.82rem;color:#fca5a5">❌ {content_preview}</div>'

            activity_html += '</div>'
            activity_placeholder.markdown(activity_html, unsafe_allow_html=True)

    # Clear live activity
    activity_placeholder.empty()

    # Update stats
    st.session_state.total_tool_calls += tool_call_count
    st.session_state.total_agents_used += len(agents_used)

    # Store assistant response
    if not final_response:
        # Fallback: use last response event
        for ev in reversed(events_log):
            if ev["event_type"] == "response":
                final_response = ev["content"]
                break

    if not final_response:
        final_response = "I couldn't generate a response. Please try again."

    st.session_state.messages.append({
        "role": "assistant",
        "content": final_response,
        "events": events_log,
        "tool_calls": tool_call_count,
        "agents": list(agents_used),
    })

    # Rerun to display
    st.rerun()


if submit and user_input and user_input.strip():
    if not st.session_state.is_processing:
        st.session_state.is_processing = True
        st.session_state.input_key += 1
        try:
            process_message(user_input.strip())
        finally:
            st.session_state.is_processing = False

elif submit and not user_input.strip():
    st.warning("⚠️ Please enter a message first!")
