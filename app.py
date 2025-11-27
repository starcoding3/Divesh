import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

from architecture_agent import (
    call_llm_for_architecture,
    validate_nfr_for_architecture,
)
from diagram_generator import generate_graphviz_diagram

app = Flask(__name__)
CORS(app)


@app.route("/api/chat", methods=["POST"])
def api_chat(): 
    data = request.json
    conversation_id = data.get("conversation_id") or str(uuid.uuid4())
    history = data.get("history", "")
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty user message."}), 400

    # Combine full requirements text
    full_requirements_text = (history + "\n" + user_message).strip()
    
    # HYBRID AGENT
    # o Workflow orchestrates steps.
    # o One LLM node with tool-calling for:
    #     - generate_architecture
    #     - render_diagram
    #     - validate_non_functionals
    # o LLM decides between tools, but workflow decides when to call LLM.

    # Step 1 — Architecture agent (using LangGraph)
    arch_plan = call_llm_for_architecture(
        full_requirements_text,
        thread_id=conversation_id
    )

    # Step 2 — Diagram generation
    image_url, dot_source = generate_graphviz_diagram(arch_plan)

    # Step 3 — NFR validation agent
    nfr_report = validate_nfr_for_architecture(
        full_requirements_text,
        arch_plan
    )

    # Final response
    response_payload = {
        "summary": arch_plan.get("summary"),
        "pattern_id": arch_plan.get("pattern_id"),
        "components": arch_plan.get("components", []),
        "connections": arch_plan.get("connections", []),

        "image_url": image_url,
        "dot": dot_source,
        
        "nfr_report": nfr_report,
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
