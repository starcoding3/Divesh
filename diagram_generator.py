"""
diagram_generator.py

Takes a structured architecture plan (components + connections)
and renders a Graphviz diagram to a PNG image.

Also returns the Graphviz DOT source so the user can edit it
or load it into other tools.
"""

import os
from typing import Dict, Any, Tuple
from uuid import uuid4
from graphviz import Digraph

# Manually add Graphviz bin folder (no need to add to system PATH)
os.environ["PATH"] += os.pathsep + r"C:\graphviz-14.0.4\bin"

def ensure_output_dir() -> str:
    """
    Ensure the 'static/diagrams' directory exists.
    Flask will serve files from the 'static' folder by default.
    """
    output_dir = os.path.join("static", "diagrams")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_graphviz_diagram(arch_plan):
    components = arch_plan.get("components", [])
    connections = arch_plan.get("connections", [])

    # Create the graph
    dot = Digraph(comment="Architecture Diagram")

    # Layout tuning for cleaner look
    dot.attr(
        "graph",
        rankdir="LR",      # left → right
        splines="ortho",   # orthogonal, right-angle edges
        concentrate="true",# merge parallel edges where possible
        nodesep="0.6",
        ranksep="0.9",
    )

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontsize="12",
    )
    dot.attr("edge", fontsize="9")

    # --- Group components by type ---
    layers = {
        "frontend": [],
        "gateway": [],
        "services": [],
        "databases": [],
        "pipeline": [],
        "other": [],
    }

    for c in components:
        ctype = (c.get("type") or "").lower()
        if ctype in ("client", "web"):
            layers["frontend"].append(c)
        elif ctype in ("gateway",):
            layers["gateway"].append(c)
        elif ctype in ("app", "service", "microservice"):
            layers["services"].append(c)
        elif ctype in ("database", "db"):
            layers["databases"].append(c)
        elif ctype in ("data_pipeline", "pipeline", "etl"):
            layers["pipeline"].append(c)
        else:
            layers["other"].append(c)

    # Helper to add a cluster if it has nodes
    def add_cluster(name, label, comps):
        if not comps:
            return
        with dot.subgraph(name=name) as sg:
            sg.attr(label=label, style="rounded,dashed", color="#cccccc")
            for c in comps:
                sg.node(c["id"], c["label"])

    # --- Create visual clusters ---
    add_cluster("cluster_frontend", "Frontend", layers["frontend"])
    add_cluster("cluster_gateway", "API Gateway", layers["gateway"])
    add_cluster("cluster_services", "Services", layers["services"])
    add_cluster("cluster_databases", "Databases", layers["databases"])
    add_cluster("cluster_pipeline", "Reporting / Data Pipeline", layers["pipeline"])
    add_cluster("cluster_other", "Other", layers["other"])

    # --- Draw edges ---
    for conn in connections:
        src = conn.get("from")
        dst = conn.get("to")
        if not src or not dst:
            continue

        label = conn.get("label") or ""

        # Option A: keep labels (a bit busier but informative)
        dot.edge(src, dst, label=label)

        # Option B (cleaner): hide labels
        # dot.edge(src, dst)

    # --- Output SVG ---
    output_dir = os.path.join("static", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    file_id = uuid4().hex
    filename = f"arch_{file_id}"
    filepath = os.path.join(output_dir, filename)

    dot.format = "svg"
    rendered_path = dot.render(filename=filepath, cleanup=True)

    # Convert filesystem path to Flask static URL
    relative_path = rendered_path.replace("\\", "/")
    dot_source = dot.source

    return "/" + relative_path, dot_source