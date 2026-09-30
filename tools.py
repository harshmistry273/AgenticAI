"""
tools.py — All tools available to the agentic AI system.
Each tool is a self-contained callable with a clear description for the LLM.
"""

import json
import math
import datetime
import traceback
import requests
import pytz
import sympy
from typing import Any
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ─────────────────────────────────────────────
#  Tool registry
# ─────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {}


def register_tool(name: str, description: str, parameters: dict):
    """Decorator to register a function as a tool."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "fn": fn,
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            },
        }
        return fn
    return decorator


# ─────────────────────────────────────────────
#  1. Web Search
# ─────────────────────────────────────────────

@register_tool(
    name="web_search",
    description=(
        "Search the web for real-time information using DuckDuckGo. "
        "Use this for current events, facts, news, or any information "
        "that requires up-to-date data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10, default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str, max_results: int = 5) -> str:
    try:
        max_results = max(1, min(10, max_results))
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        if not results:
            return json.dumps({"error": "No results found", "query": query})
        return json.dumps({"query": query, "results": results}, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


# ─────────────────────────────────────────────
#  2. Webpage Reader
# ─────────────────────────────────────────────

@register_tool(
    name="read_webpage",
    description=(
        "Fetch and extract the main text content from a webpage URL. "
        "Use after web_search when you need the full content of a specific page."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch"},
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 3000)",
                "default": 3000,
            },
        },
        "required": ["url"],
    },
)
def read_webpage(url: str, max_chars: int = 3000) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AgenticBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse excessive blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)[:max_chars]
        return json.dumps({"url": url, "content": text})
    except Exception as e:
        return json.dumps({"error": str(e), "url": url})


# ─────────────────────────────────────────────
#  3. Calculator / Math
# ─────────────────────────────────────────────

@register_tool(
    name="calculator",
    description=(
        "Evaluate mathematical expressions and equations. Supports arithmetic, "
        "algebra, trigonometry, calculus (derivatives/integrals), and symbolic math. "
        "Examples: '2**32', 'sin(pi/4)', 'diff(x**2, x)', 'integrate(x**2, x)', "
        "'solve(x**2 - 4, x)'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression or equation to evaluate",
            },
            "mode": {
                "type": "string",
                "enum": ["evaluate", "simplify", "solve", "diff", "integrate"],
                "description": "Mode of operation (default: evaluate)",
                "default": "evaluate",
            },
            "variable": {
                "type": "string",
                "description": "Variable for solve/diff/integrate (default: x)",
                "default": "x",
            },
        },
        "required": ["expression"],
    },
)
def calculator(expression: str, mode: str = "evaluate", variable: str = "x") -> str:
    try:
        x, y, z, n, t, a, b, c = sympy.symbols("x y z n t a b c")
        sym_vars = {"x": x, "y": y, "z": z, "n": n, "t": t, "a": a, "b": b, "c": c}
        
        # Safe evaluation namespace
        ns = {
            **sym_vars,
            **{k: getattr(sympy, k) for k in dir(sympy) if not k.startswith("_")},
            "pi": sympy.pi, "e": sympy.E,
        }

        var = sym_vars.get(variable, x)
        expr = sympy.sympify(expression, locals=ns)

        if mode == "solve":
            result = sympy.solve(expr, var)
        elif mode == "diff":
            result = sympy.diff(expr, var)
        elif mode == "integrate":
            result = sympy.integrate(expr, var)
        elif mode == "simplify":
            result = sympy.simplify(expr)
        else:
            result = sympy.N(expr) if expr.is_number else sympy.simplify(expr)

        return json.dumps({
            "expression": expression,
            "mode": mode,
            "result": str(result),
            "latex": sympy.latex(result) if hasattr(result, "__iter__") is False else str(result),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


# ─────────────────────────────────────────────
#  4. Date & Time
# ─────────────────────────────────────────────

@register_tool(
    name="get_datetime",
    description=(
        "Get the current date and time, optionally for a specific timezone. "
        "Can also compute date differences or add/subtract days."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'Asia/Kolkata', 'US/Eastern', 'UTC'). Default: UTC",
                "default": "UTC",
            },
            "operation": {
                "type": "string",
                "enum": ["current", "add_days", "diff"],
                "description": "Operation to perform",
                "default": "current",
            },
            "days": {
                "type": "integer",
                "description": "Number of days to add (for add_days operation)",
                "default": 0,
            },
            "date1": {
                "type": "string",
                "description": "First date (YYYY-MM-DD) for diff operation",
            },
            "date2": {
                "type": "string",
                "description": "Second date (YYYY-MM-DD) for diff operation",
            },
        },
        "required": [],
    },
)
def get_datetime(
    timezone: str = "UTC",
    operation: str = "current",
    days: int = 0,
    date1: str = None,
    date2: str = None,
) -> str:
    try:
        tz = pytz.timezone(timezone)
        now = datetime.datetime.now(tz)

        if operation == "add_days":
            target = now + datetime.timedelta(days=days)
            return json.dumps({
                "timezone": timezone,
                "base_date": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "days_added": days,
                "result_date": target.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "weekday": target.strftime("%A"),
            })
        elif operation == "diff" and date1 and date2:
            d1 = datetime.date.fromisoformat(date1)
            d2 = datetime.date.fromisoformat(date2)
            delta = abs((d2 - d1).days)
            return json.dumps({
                "date1": date1, "date2": date2,
                "difference_days": delta,
                "difference_weeks": round(delta / 7, 2),
                "difference_months": round(delta / 30.44, 2),
                "difference_years": round(delta / 365.25, 2),
            })
        else:
            return json.dumps({
                "timezone": timezone,
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "weekday": now.strftime("%A"),
                "week_number": now.isocalendar()[1],
                "timestamp": int(now.timestamp()),
                "utc_offset": now.strftime("%z"),
            })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────
#  5. Wikipedia Summary
# ─────────────────────────────────────────────

@register_tool(
    name="wikipedia_search",
    description=(
        "Search Wikipedia and get a summary of a topic. "
        "Ideal for factual, encyclopedic information about people, places, concepts, history, science, etc."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The topic to search on Wikipedia"},
            "sentences": {
                "type": "integer",
                "description": "Number of sentences to return (default: 5, max: 15)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
def wikipedia_search(query: str, sentences: int = 5) -> str:
    try:
        sentences = max(1, min(15, sentences))
        # Use Wikipedia's REST API (no auth needed)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}"
        headers = {"User-Agent": "AgenticAI/1.0 (educational project)"}
        resp = requests.get(url, headers=headers, timeout=10)

        if resp.status_code == 404:
            # Try search API
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "opensearch", "search": query,
                "limit": 3, "format": "json",
            }
            sr = requests.get(search_url, params=params, headers=headers, timeout=10)
            titles = sr.json()[1]
            if titles:
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(titles[0])}"
                resp = requests.get(url, headers=headers, timeout=10)
            else:
                return json.dumps({"error": f"No Wikipedia article found for '{query}'"})

        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "")
        # Limit sentences
        import re
        sentence_list = re.split(r'(?<=[.!?])\s+', extract)
        summary = " ".join(sentence_list[:sentences])

        return json.dumps({
            "title": data.get("title", ""),
            "summary": summary,
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "thumbnail": data.get("thumbnail", {}).get("source", ""),
        })
    except Exception as e:
        return json.dumps({"error": str(e), "query": query})


# ─────────────────────────────────────────────
#  6. Code Execution (safe Python sandbox)
# ─────────────────────────────────────────────

@register_tool(
    name="execute_python",
    description=(
        "Execute Python code in a sandboxed environment. "
        "Use for data analysis, computation, generating results, creating charts data, etc. "
        "Available libraries: math, json, datetime, collections, itertools, functools, "
        "statistics, random, re, string, sympy, pandas (basic). "
        "Returns stdout output and any errors."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute",
            },
        },
        "required": ["code"],
    },
)
def execute_python(code: str) -> str:
    import io
    import sys
    import math
    import json as _json
    import random
    import statistics
    import re
    import string
    import collections
    import itertools
    import functools
    import datetime as dt

    # Safe globals whitelist
    safe_globals = {
        "__builtins__": {
            "print": print, "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "reversed": reversed, "list": list, "dict": dict, "set": set,
            "tuple": tuple, "str": str, "int": int, "float": float, "bool": bool,
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "any": any, "all": all, "isinstance": isinstance, "type": type,
            "repr": repr, "format": format, "hex": hex, "oct": oct, "bin": bin,
            "ord": ord, "chr": chr, "hash": hash, "id": id,
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
        },
        "math": math, "json": _json, "random": random, "statistics": statistics,
        "re": re, "string": string, "collections": collections,
        "itertools": itertools, "functools": functools, "datetime": dt,
        "sympy": sympy,
    }

    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        exec(code, safe_globals)
        output = sys.stdout.getvalue()
        return json.dumps({
            "status": "success",
            "output": output if output else "(no output)",
            "code_lines": len(code.strip().splitlines()),
        })
    except Exception:
        error = traceback.format_exc()
        return json.dumps({"status": "error", "error": error})
    finally:
        sys.stdout = old_stdout


# ─────────────────────────────────────────────
#  7. Unit Converter
# ─────────────────────────────────────────────

@register_tool(
    name="unit_converter",
    description=(
        "Convert between units of measurement. "
        "Supports: length, weight/mass, temperature, area, volume, speed, data/storage."
    ),
    parameters={
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The numeric value to convert"},
            "from_unit": {"type": "string", "description": "Source unit (e.g., 'km', 'kg', 'celsius', 'GB')"},
            "to_unit": {"type": "string", "description": "Target unit (e.g., 'miles', 'lbs', 'fahrenheit', 'MB')"},
            "category": {
                "type": "string",
                "enum": ["length", "weight", "temperature", "area", "volume", "speed", "data"],
                "description": "Category of conversion",
            },
        },
        "required": ["value", "from_unit", "to_unit", "category"],
    },
)
def unit_converter(value: float, from_unit: str, to_unit: str, category: str) -> str:
    try:
        fu = from_unit.lower().strip()
        tu = to_unit.lower().strip()

        # Conversion tables (to SI base)
        conversions = {
            "length": {
                "m": 1, "km": 1000, "cm": 0.01, "mm": 0.001,
                "miles": 1609.344, "mile": 1609.344, "ft": 0.3048,
                "feet": 0.3048, "inch": 0.0254, "inches": 0.0254,
                "yard": 0.9144, "yards": 0.9144, "nm": 1852,
            },
            "weight": {
                "kg": 1, "g": 0.001, "mg": 0.000001, "lbs": 0.453592,
                "lb": 0.453592, "oz": 0.028349, "ton": 1000, "tonne": 1000,
                "stone": 6.35029,
            },
            "area": {
                "m2": 1, "km2": 1e6, "cm2": 0.0001, "ft2": 0.092903,
                "acre": 4046.86, "hectare": 10000, "miles2": 2.59e6,
            },
            "volume": {
                "l": 1, "ml": 0.001, "m3": 1000, "gallon": 3.78541,
                "quart": 0.946353, "pint": 0.473176, "cup": 0.236588,
                "fl_oz": 0.0295735,
            },
            "speed": {
                "m/s": 1, "km/h": 1/3.6, "mph": 0.44704,
                "knot": 0.514444, "ft/s": 0.3048,
            },
            "data": {
                "b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3,
                "tb": 1024**4, "pb": 1024**5,
            },
        }

        if category == "temperature":
            if fu == "celsius" and tu == "fahrenheit":
                result = value * 9/5 + 32
            elif fu == "fahrenheit" and tu == "celsius":
                result = (value - 32) * 5/9
            elif fu == "celsius" and tu == "kelvin":
                result = value + 273.15
            elif fu == "kelvin" and tu == "celsius":
                result = value - 273.15
            elif fu == "fahrenheit" and tu == "kelvin":
                result = (value - 32) * 5/9 + 273.15
            elif fu == "kelvin" and tu == "fahrenheit":
                result = (value - 273.15) * 9/5 + 32
            else:
                result = value
        else:
            table = conversions.get(category, {})
            if fu not in table or tu not in table:
                return json.dumps({
                    "error": f"Unknown units '{from_unit}' or '{to_unit}' for {category}",
                    "available": list(table.keys()),
                })
            result = value * table[fu] / table[tu]

        return json.dumps({
            "input": f"{value} {from_unit}",
            "output": f"{round(result, 6)} {to_unit}",
            "result": round(result, 6),
            "category": category,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ─────────────────────────────────────────────
#  Helper: get all tool schemas for Groq
# ─────────────────────────────────────────────

def get_tool_schemas() -> list[dict]:
    return [info["schema"] for info in TOOL_REGISTRY.values()]


def execute_tool(name: str, arguments: dict) -> str:
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return TOOL_REGISTRY[name]["fn"](**arguments)
    except Exception as e:
        return json.dumps({"error": str(e), "tool": name, "args": arguments})
