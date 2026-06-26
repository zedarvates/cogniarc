#!/usr/bin/env python3
"""
CogniARC MCP Bridge — Hermes Agent cognitive reasoning server.

Exposes CogniARC's cognitive drives + Sternberg's triarchic model as MCP tools
that Hermes agents can query for decision-making, problem representation,
and cognitive balance monitoring.

Protocol: JSON-RPC 2.0 over stdio (MCP standard)

Tools exposed:
  cogniarc_think     — Evaluate a decision using all 6 cognitive drives
  cogniarc_represent — Generate Sternberg's 3 representations of a problem
  cogniarc_balance   — Get current triarchic balance + recommendation
  cogniarc_reframe   — Get a reframing question when stuck (Douglass pattern)

Usage:
  python cogniarc_mcp_bridge.py
  # Or register as MCP server in Hermes: hermes mcp add cogniarc --command "python cogniarc/..." 
"""
import sys, json
from typing import Any

from cogniarc.cognitive_player import CognitiveDrives, hash_grid
from cogniarc.triarchic_engine import TriarchicEngine
from cogniarc.representation_engine import RepresentationEngine


# Global state (one instance per server process)
drives = CognitiveDrives()
triarchic = TriarchicEngine()
representer = RepresentationEngine()

TOOLS = {
    "cogniarc_think": {
        "description": "Evaluate a decision using all 6 cognitive drives. Returns scored recommendation.",
        "parameters": {
            "decision": "The decision or action to evaluate (e.g., 'audit this code')",
            "context": "Optional: what's the current situation? (e.g., 'stuck in a loop')",
            "constraints": "Optional: any constraints? (e.g., 'must be local, no cloud')",
        }
    },
    "cogniarc_represent": {
        "description": "Generate Sternberg's 3 representations of a problem (causal chain, dependency graph, core question). Based on Faraday's notebook method.",
        "parameters": {
            "problem": "The problem to represent (e.g., 'the build is broken')",
        }
    },
    "cogniarc_balance": {
        "description": "Get current triarchic balance (analytical/creative/practical) + recommendation on which mode to activate.",
        "parameters": {}
    },
    "cogniarc_reframe": {
        "description": "Get a reframing question when stuck. Based on Frederick Douglass's pattern: question the frame, not the argument.",
        "parameters": {}
    },
}


def handle_list_tools() -> dict:
    return {
        "tools": [
            {"name": name, "description": info["description"],
             "inputSchema": {"type": "object", "properties": {
                 k: {"type": "string", "description": v} for k, v in info["parameters"].items()
             }}}
            for name, info in TOOLS.items()
        ]
    }


def handle_call_tool(name: str, arguments: dict) -> dict:
    # Validate required arguments per tool
    if name == "cogniarc_think":
        if not arguments.get("decision"):
            return {"error": "cogniarc_think requires 'decision' argument"}
        return cogniarc_think(arguments.get("decision", ""),
                            arguments.get("context", ""),
                            arguments.get("constraints", ""))
    elif name == "cogniarc_represent":
        if not arguments.get("problem"):
            return {"error": "cogniarc_represent requires 'problem' argument"}
        return cogniarc_represent(arguments.get("problem", ""))
    elif name == "cogniarc_balance":
        return cogniarc_balance()
    elif name == "cogniarc_reframe":
        return cogniarc_reframe()
    return {"error": f"Unknown tool: {name}"}


def cogniarc_think(decision: str, context: str = "", constraints: str = "") -> dict:
    """Score a decision through all 6 cognitive drives + triarchic balance."""
    state_hash = hash_grid(None) if not context else str(hash(context.encode()))

    # Score with classic drives
    action_scores = {}
    for action_id in range(1, 5):
        action_scores[action_id] = round(
            drives.score_action(action_id, state_hash, plan_length=1), 3
        )

    # Update triarchic balance
    drive_scores = {
        'novelty': drives.novelty_score(state_hash),
        'simplicity': drives.simplicity_score(1),
        'doubt': drives.doubt_score(),
        'pleasure': drives.pleasure_score(None),
        'caution': drives.caution_score(1),
        'impulse': drives.impulse_score(),
    }
    triarchic.update(drive_scores)

    # Build recommendation
    reco = triarchic.last_recommendation
    balance_info = {
        "balance": round(triarchic.state.balance_score, 3),
        "dominant": triarchic.state.dominant_mode,
        "weakest": triarchic.state.weakest_mode,
        "recommendation": reco,
    }

    # Check if reframe needed
    needs_reframe = triarchic.needs_reframe()

    return {
        "decision": decision,
        "cognitive_scores": drive_scores,
        "triarchic_balance": balance_info,
        "needs_reframe": needs_reframe,
        "reframe_prompt": triarchic.reframe_prompt() if needs_reframe else None,
        "verdict": _verdict(drive_scores, balance_info),
    }


def cogniarc_represent(problem: str) -> dict:
    """Generate Sternberg's 3 representations."""
    rep = representer.represent(problem)
    return rep.to_dict()


def cogniarc_balance() -> dict:
    """Get triarchic balance status."""
    return {
        "analytical": round(triarchic.state.analytical, 3),
        "creative": round(triarchic.state.creative, 3),
        "practical": round(triarchic.state.practical, 3),
        "balance": round(triarchic.state.balance_score, 3),
        "dominant_mode": triarchic.state.dominant_mode,
        "weakest_mode": triarchic.state.weakest_mode,
        "recommendation": triarchic.last_recommendation or triarchic.state.recommendation(),
        "stagnation_counter": triarchic.stagnation_counter,
        "needs_reframe": triarchic.needs_reframe(),
    }


def cogniarc_reframe() -> dict:
    """Get a reframing question."""
    triarchic.stagnation_counter += 1  # Force reframe consideration
    return {
        "reframe_question": triarchic.reframe_prompt(),
        "pattern": "Douglass: question the frame, not the argument",
        "triarchic_balance": round(triarchic.state.balance_score, 3),
    }


def _verdict(scores: dict, balance: dict) -> str:
    """Synthesize a human-readable verdict."""
    if balance["balance"] > 0.7:
        return f"✅ Balanced decision. {balance['recommendation']}"
    elif balance["balance"] > 0.4:
        return f"⚠️ Slightly imbalanced — boost {balance['weakest']} mode. {balance['recommendation']}"
    else:
        return f"🔴 Imbalanced! Dominant: {balance['dominant']}, missing: {balance['weakest']}. {balance['recommendation']}"


def main():
    """MCP JSON-RPC loop over stdio."""
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break

            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method", "")

            if method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": handle_list_tools()}
            elif method == "tools/call":
                params = request.get("params", {})
                result = handle_call_tool(params.get("name", ""), params.get("arguments", {}))
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": json.dumps(result)}]}}
            elif method == "initialize":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "cogniarc-mcp", "version": "1.0.0"},
                    "capabilities": {"tools": {}}
                }}
            else:
                response = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            continue
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
