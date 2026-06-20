#!/usr/bin/env python3
"""
CogniARC CLI — Unified interface for cognitive reasoning
Connects to: Hermes Rust Backend (LLM), CogniARC MCP (drives), LocalAI (fallback)

Usage:
  cogniarc think "Should I deploy now?"         # Evaluate decision via 6 drives
  cogniarc represent "This bug is weird"         # 3 representations (Faraday)
  cogniarc balance                               # Check triarchic balance
  cogniarc reframe "I'm stuck"                   # Reframing question (Douglass)
  cogniarc chat "Explain quantum computing"      # Direct LLM query via Rust backend
  cogniarc engines                               # Check all engine statuses
  cogniarc analyze "problem description"      # Full: think + represent + balance
  cogniarc vision https://example.com           # Visual analysis (browser-use + Hailo-8)
  cogniarc watch https://example.com            # Watch page for changes
"""
import argparse, json, sys, os, subprocess, textwrap, datetime
from pathlib import Path
from typing import Optional

# Default endpoints
RUST_BACKEND = os.environ.get("COGNIARC_RUST_URL", "http://192.168.1.47:8769")
LOCALAI_URL = os.environ.get("COGNIARC_LOCALAI_URL", "http://192.168.1.47:8080")
COGNIARC_PROJECT = Path(__file__).parent.parent

def api_post(url: str, payload: dict, timeout: int = 30) -> dict:
    """Make HTTP POST request compatible with both requests and curl"""
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}

def api_get(url: str, timeout: int = 10) -> dict:
    import urllib.request
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

# ── Commands ─────────────────────────────────────────────────

def cmd_think(args):
    """Evaluate a decision using the 6 drives via Rust backend"""
    print(f"🧠 CogniARC — Thinking about: \"{args.text}\"\n")
    
    # Try Rust backend first
    result = api_post(f"{RUST_BACKEND}/v1/chat/completions", {
        "model": "cogniarc-think",
        "messages": [
            {"role": "system", "content": "You are CogniARC, a cognitive reasoning engine. "
             "Evaluate the user's decision using these 6 drives: "
             "1) Curiosity (what's unknown?) 2) Pattern Match (similar past situations?) "
             "3) Causal (what are the consequences?) 4) Efficiency (simplest path?) "
             "5) Memory (what have we learned?) 6) Verify (what could go wrong?). "
             "Return JSON: {drives: [{name, score: 0-10, reason}], recommendation, confidence: 0-1}"},
            {"role": "user", "content": args.text}
        ],
        "max_tokens": 1024,
        "temperature": 0.3
    })
    
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if content:
        # Try to extract JSON
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for d in data.get("drives", []):
                    bar = "█" * d.get("score", 0) + "░" * (10 - d.get("score", 0))
                    print(f"  {d.get('name', '?'):15} [{bar}] {d.get('score', 0):2d}/10  — {d.get('reason', '')}")
                print(f"\n  💡 Recommendation: {data.get('recommendation', 'N/A')}")
                print(f"  📊 Confidence: {data.get('confidence', 0):.0%}")
            else:
                print(f"  {content[:500]}")
        except:
            print(f"  {content[:500]}")
    else:
        print(f"  ⚠️ No response from backend: {result.get('error', 'unknown')}")
    
    print()

def cmd_represent(args):
    """Generate 3 representations (Faraday method)"""
    print(f"🔄 3 Representations of: \"{args.text}\"\n")
    
    result = api_post(f"{RUST_BACKEND}/v1/chat/completions", {
        "model": "cogniarc-represent",
        "messages": [
            {"role": "system", "content": "Generate 3 representations of the problem "
             "(Faraday method):\n1) Causal Chain (cause→effect→outcome)\n"
             "2) Dependency Graph (what depends on what)\n"
             "3) Fundamental Question (the core tension)\n"
             "Return as JSON with keys: causal_chain, dependency_graph, fundamental_question"},
            {"role": "user", "content": args.text}
        ],
        "max_tokens": 1024,
        "temperature": 0.4
    })
    
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if content:
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                print(f"  🔗 Causal Chain:      {data.get('causal_chain', 'N/A')[:200]}")
                print(f"  🕸️ Dependency Graph:  {data.get('dependency_graph', 'N/A')[:200]}")
                print(f"  ❓ Key Question:      {data.get('fundamental_question', 'N/A')[:200]}")
            else:
                print(f"  {content[:500]}")
        except:
            print(f"  {content[:500]}")
    else:
        print(f"  ⚠️ {result.get('error', 'No response')}")

def cmd_balance(args):
    """Check triarchic balance via engines"""
    print("⚖️ CogniARC — Triarchic Balance Check\n")
    engines = api_get(f"{RUST_BACKEND}/engines")
    
    gguf = engines.get("engines", {}).get("gguf", {})
    hailo = engines.get("engines", {}).get("hailo8", {})
    cuda = engines.get("engines", {}).get("cuda", {})
    
    print(f"  🧠 GGUF (Analytical):   {'✅' if gguf.get('available') else '❌'} {gguf.get('backend', '?')} ({gguf.get('gpu_count', 0)} GPU)")
    print(f"  👁️  Hailo-8 (Creative):  {'✅' if hailo.get('available') else '❌'} device: {hailo.get('device', 'N/A')}")
    print(f"  🖥️  CUDA (Practical):    {'✅' if cuda.get('available') else '❌'} {cuda.get('device_count', 0)} devices")
    
    if gguf.get("available") and hailo.get("available") and cuda.get("device_count", 0) > 0:
        print(f"\n  ✅ Triarchic balance: STABLE (all 3 engines available)")
    else:
        print(f"\n  ⚠️ Triarchic balance: DEGRADED (some engines unavailable)")
    
    # Show available models
    models = gguf.get("models", [])
    if models:
        print(f"\n  📦 Models ({len(models)}):")
        for m in models[:5]:
            print(f"     - {m}")

def cmd_reframe(args):
    """Get a reframing question (Douglass method)"""
    result = api_post(f"{RUST_BACKEND}/v1/chat/completions", {
        "model": "cogniarc-reframe",
        "messages": [
            {"role": "system", "content": "You are using the Douglass reframing method. "
             "Question the frame, not the argument. Generate a single question that "
             "completely reframes the user's situation. Return ONLY the question."},
            {"role": "user", "content": args.text}
        ],
        "max_tokens": 256,
        "temperature": 0.8
    })
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"\n❓ Reframing Question:\n\n  \"{content}\"\n" if content else f"\n  ⚠️ No response\n")

def cmd_chat(args):
    """Direct chat via Rust backend"""
    result = api_post(f"{RUST_BACKEND}/v1/chat/completions", {
        "model": args.model or "gpt-4o",
        "messages": [{"role": "user", "content": args.text}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    })
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if content:
        print(f"\n  {content}\n")
    else:
        print(f"\n  ⚠️ {result.get('error', 'No response')}\n")

def cmd_engines(args):
    """Status of all connected engines"""
    print("🔌 CogniARC — Engine Status\n")
    
    # Rust backend
    health = api_get(f"{RUST_BACKEND}/health")
    print(f"  🦀 Rust Backend ({RUST_BACKEND}):")
    print(f"     Status:  {'✅' if health.get('status') == 'ok' else '❌'}")
    print(f"     Version: {health.get('version', '?')}")
    print(f"     Features: {', '.join(health.get('features', []))}")
    
    # Engines detail
    engines = api_get(f"{RUST_BACKEND}/engines")
    for name, info in engines.get("engines", {}).items():
        status = "✅" if info.get("available") else "❌"
        details = ", ".join(f"{k}={v}" for k, v in info.items() if k != "available")
        print(f"     {status} {name}: {details}")
    
    # LocalAI direct
    la = api_get(f"{LOCALAI_URL}/v1/models")
    if "error" not in la:
        models = la.get("data", [])
        print(f"\n  🧠 LocalAI ({LOCALAI_URL}): ✅ ({len(models)} models)")
    else:
        print(f"\n  🧠 LocalAI ({LOCALAI_URL}): ❌")

def cmd_analyze(args):
    """Full analysis: think + represent + balance"""
    print(f"\n{'═'*60}")
    print(f"  📋 CogniARC — Full Analysis")
    print(f"  Problem: {args.text}")
    print(f"{'═'*60}\n")
    
    cmd_think(args)
    cmd_represent(args)
    cmd_balance(args)

# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CogniARC — Cognitive Reasoning CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              cogniarc think "Should I deploy this now?"
              cogniarc represent "The UI bug appears randomly"
              cogniarc balance
              cogniarc reframe "I can't solve this puzzle"
              cogniarc chat "Explain attention mechanisms"
              cogniarc analyze "The database is slow"
              cogniarc engines
        """))
    
    sub = parser.add_subparsers(dest="command")
    
    p_think = sub.add_parser("think", help="Evaluate a decision via 6 drives")
    p_think.add_argument("text", help="Decision to evaluate")
    
    p_repr = sub.add_parser("represent", help="3 representations (Faraday)")
    p_repr.add_argument("text", help="Problem to analyze")
    
    p_bal = sub.add_parser("balance", help="Check triarchic balance")
    
    p_reframe = sub.add_parser("reframe", help="Reframing question (Douglass)")
    p_reframe.add_argument("text", help="Situation to reframe")
    
    p_chat = sub.add_parser("chat", help="Direct LLM query")
    p_chat.add_argument("text", help="Your question")
    p_chat.add_argument("--model", "-m", default="gpt-4o", help="Model name")
    p_chat.add_argument("--max-tokens", type=int, default=512)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    
    p_eng = sub.add_parser("engines", help="Engine status")
    
    p_analyze = sub.add_parser("analyze", help="Full analysis (all modes)")
    p_analyze.add_argument("text", help="Problem to analyze fully")
    
    p_vision = sub.add_parser("vision", help="Visual analysis via browser-use + Hailo-8")
    p_vision.add_argument("url", help="URL to analyze visually")
    
    p_watch = sub.add_parser("watch", help="Watch a page for changes")
    p_watch.add_argument("url", help="URL to watch")
    p_watch.add_argument("--interval", "-i", type=int, default=60, help="Check interval (seconds)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "think": cmd_think,
        "represent": cmd_represent,
        "balance": cmd_balance,
        "reframe": cmd_reframe,
        "chat": cmd_chat,
        "engines": cmd_engines,
        "analyze": cmd_analyze,
        "vision": lambda a: __import__('cogniarc.vision_sensor', fromlist=['cmd_analyze']).cmd_analyze(a.url),
        "watch": lambda a: __import__('cogniarc.vision_sensor', fromlist=['cmd_watch']).cmd_watch(a.url, a.interval),
    }
    
    commands[args.command](args)

if __name__ == "__main__":
    main()
