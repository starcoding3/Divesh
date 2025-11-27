import json
from typing import Dict, Any, List, TypedDict, Annotated
import httpx
from langchain_openai import ChatOpenAI
import config
import logging
import traceback
from openai import InternalServerError
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ===========================
# AZURE OPENAI CLIENT
# ===========================

http_client = httpx.Client(verify=False)

client = ChatOpenAI(
    api_key=config.AZURE_OPENAI_API_KEY,
    base_url=config.AZURE_OPENAI_ENDPOINT,
    model=config.OPENAI_MODEL,
    http_client=http_client,
    temperature=0.1,
)

# ===========================
# Load templates
# ===========================

with open(config.TEMPLATES_PATH, "r", encoding="utf-8") as f:
    TEMPLATE_DATA = json.load(f)


# ======================================================
# Build prompt for architecture design + refinement
# ======================================================

def build_prompt_messages(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> List[Dict[str, Any]]:
    template_summaries = [
        {"id": p["id"], "name": p["name"], "description": p["description"]}
        for p in TEMPLATE_DATA.get("patterns", [])
    ]
    templates_str = json.dumps(template_summaries, indent=2)

    system_content = (
        "You are an Architecture Design Assistant for IT systems. "
        "Your job is to take high-level requirements and propose a system architecture.\n\n"
        "You have access to a library of architecture patterns. "
        "Return ONLY valid JSON with keys: summary (HTML), pattern_id, components, connections.\n"
        "Use stable component IDs. Refine previous architecture if provided.\n"
    )

    user_parts = []
    user_parts.append("Available architecture patterns:\n")
    user_parts.append(templates_str)

    if previous_arch_plan:
        user_parts.append("\n\nPrevious architecture plan (baseline):\n")
        user_parts.append(json.dumps(previous_arch_plan, indent=2))
        user_parts.append("\n\nUser refinement request:\n")
        user_parts.append(user_message)
    else:
        user_parts.append("\n\nFull user requirements:\n")
        user_parts.append(user_message)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "".join(user_parts)},
    ]
    return messages


# ======================================================
# Call architecture model
# ======================================================

def _call_model(
    user_message: str,
    previous_arch_plan: Dict[str, Any] | None,
) -> Dict[str, Any]:
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    messages = build_prompt_messages(user_message, previous_arch_plan)

    system_content = messages[0]["content"]
    user_content = messages[1]["content"]
    full_prompt = system_content + "\n\n" + user_content

    try:
        llm_result = client.invoke(full_prompt)
        raw_text = getattr(llm_result, "content", str(llm_result))

        clean = raw_text.strip()
        if clean.startswith("```"):
            import re
            match = re.search(r"\{[\s\S]*\}", clean)
            if match:
                clean = match.group(0)

        try:
            arch_plan = json.loads(clean)
        except Exception as e:
            arch_plan = _fallback_architecture("Invalid JSON from model.")

    except InternalServerError as e:
        raise RuntimeError("Azure gateway 500 — check logs.") from e

    except Exception as ex:
        raise RuntimeError("Connection failure to Azure OpenAI.") from ex

    arch_plan.setdefault("summary", "No summary provided.")
    arch_plan.setdefault("pattern_id", "unknown")
    arch_plan.setdefault("components", [])
    arch_plan.setdefault("connections", [])

    return arch_plan


def _fallback_architecture(reason: str) -> Dict[str, Any]:
    return {
        "summary": f"Fallback architecture: {reason}",
        "pattern_id": "fallback_three_tier",
        "components": [
            {"id": "client", "label": "Client", "type": "client"},
            {"id": "web", "label": "Web Server", "type": "web"},
            {"id": "app", "label": "App Server", "type": "app"},
            {"id": "db", "label": "Database", "type": "database"},
        ],
        "connections": [
            {"from": "client", "to": "web", "label": "HTTP"},
            {"from": "web", "to": "app", "label": "Internal HTTP"},
            {"from": "app", "to": "db", "label": "SQL"},
        ],
    }


# ======================================================
# LangGraph state + workflow (unchanged)
# ======================================================

class ArchState(TypedDict):
    messages: Annotated[List[str], operator.add]
    arch_plan: Dict[str, Any]
    arch_history: Annotated[List[Dict[str, Any]], operator.add]


# This is a node
def _llm_node(state: ArchState) -> ArchState:
    # 1. node receives state
    msgs = state.get("messages") or []
    latest_req = msgs[-1]

    hist = state.get("arch_history") or []
    previous_arch = hist[-1] if hist else None
    
    # 2. node does its job
    arch_plan = _call_model(latest_req, previous_arch)

    # 3. node returns the updated state
    return {
        "messages": [],
        "arch_plan": arch_plan,
        "arch_history": [arch_plan],
    }


# Here we are creating a graph using LangGraph. This is a one node graph:
#                          +---------------+
# START --> LOAD STATE --> |   _llm_node   | --> SAVE STATE IN MEMORY --> END
#                          +---------------+
# We could have created a multi-node graph like: [LLM Node] → [NFR Node] → [Diagram Node] → END
# But we kept only 1 node inside this graph to preserve stability and calling NFR + Diagram outside for simplicity. 
_graph_builder = StateGraph(ArchState)
_graph_builder.add_node("llm", _llm_node)
_graph_builder.set_entry_point("llm")
_graph_builder.add_edge("llm", END)

_checkpointer = MemorySaver()
# Takes the graph definition (nodes + edges + memory) and converts it into a workflow engine.
# Graph is a "blueprint" and Workflow is the "running engine"
_arch_graph = _graph_builder.compile(checkpointer=_checkpointer)


def call_llm_for_architecture(user_message: str, thread_id: str = "default"):
    if not config.AZURE_OPENAI_API_KEY:
        raise RuntimeError("Missing Azure OpenAI API key in config.py")

    initial_state: ArchState = {
        "messages": [user_message],
        "arch_plan": {},
        "arch_history": [],
    }

    # Run the workflow engine created above
    final = _arch_graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )

    plan = final.get("arch_plan") or _fallback_architecture("Missing plan.")
    return plan


# ======================================================
# NFR VALIDATION AGENT (new)
# ======================================================

def _call_nfr_model(requirements_text: str, arch_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calls GPT-4o to produce NFR analysis.
    """
    system_prompt = (
        "You are an IT Architecture NFR reviewer. Your job is to analyze "
        "the architecture against standard NFR categories:\n"
        "- Performance\n- Scalability\n- Availability\n- Reliability\n"
        "- Security\n- Maintainability\n- Observability\n\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        "  \"overall_risk\": \"Low|Medium|High\",\n"
        "  \"summary\": \"short text\",\n"
        "  \"issues\": [\n"
        "     {\"category\": \"...\", \"severity\": \"Low|Medium|High\", "
        "      \"finding\": \"...\", \"recommendation\": \"...\"}\n"
        "  ]\n"
        "}\n"
    )

    user_prompt = (
        "Full Requirements:\n" + requirements_text + "\n\n"
        "Architecture Plan JSON:\n" + json.dumps(arch_plan, indent=2)
    )

    full_prompt = system_prompt + "\n\n" + user_prompt

    try:
        llm_result = client.invoke(full_prompt)
        text = getattr(llm_result, "content", str(llm_result)).strip()

        if text.startswith("```"):
            import re
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                text = m.group(0)

        result = json.loads(text)
        return result

    except Exception as e:
        logger.error("NFR validation failed: %s", e)
        return {
            "overall_risk": "Unknown",
            "summary": "NFR validation failed.",
            "issues": [],
        }


def validate_nfr_for_architecture(
    full_requirements_text: str,
    arch_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Public wrapper for NFR agent.
    """
    if not arch_plan:
        return {
            "overall_risk": "Unknown",
            "summary": "No architecture plan to validate.",
            "issues": [],
        }

    return _call_nfr_model(full_requirements_text, arch_plan)
