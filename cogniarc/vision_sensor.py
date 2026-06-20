#!/usr/bin/env python3
"""
CogniARC Vision Sensor — Browser-Use + Hailo-8 + Raisonnement Cognitive

Pipeline:
  1. Browser-Use navigue vers une URL
  2. Capture screenshot + extrait le texte visible
  3. Hailo-8 analyse le screenshot (YOLO UI detection + OCR)
  4. CogniARC raisonne via les 6 drives + le DAG context
  
Usage:
  python3 -m cogniarc.vision_sensor analyze https://example.com
  python3 -m cogniarc.vision_sensor think "Vérifie le bouton login" --url https://app.example.com
  python3 -m cogniarc.vision_sensor watch https://site.com --interval 60
"""
import asyncio, json, sys, os, tempfile, base64, io, re, textwrap
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import urllib.request, urllib.error

# ── Configuration ─────────────────────────────────────────────

RUST_BACKEND = os.environ.get("COGNIARC_RUST_URL", "http://192.168.1.47:8769")
HAILO_URL = os.environ.get("HAILO_API_URL", "http://192.168.1.47:8767")
LOCALAI_URL = os.environ.get("COGNIARC_LOCALAI_URL", "http://192.168.1.47:8080")
BROWSER_USE_CMD = os.environ.get("BROWSER_USE_CMD", 
    "wsl -d Ubuntu-24.04 -- bash -c 'cd /tmp && source ~/.cargo/env 2>/dev/null; OLLAMA_HOST=http://192.168.1.47:11434 python3 -c \"{}\"'")

# ── Hailo-8 Vision ───────────────────────────────────────────

def hailo_health() -> bool:
    """Check if Hailo-8 is responsive"""
    try:
        resp = urllib.request.urlopen(f"{HAILO_URL}/health", timeout=3)
        return resp.status == 200
    except:
        return False

def hailo_detect(image_path: str) -> Optional[dict]:
    """Detect UI elements in screenshot via Hailo-8 YOLOv8m"""
    try:
        data = json.dumps({"image_path": image_path}).encode()
        req = urllib.request.Request(f"{HAILO_URL}/vision/detect", data=data,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def hailo_ocr(image_path: str) -> Optional[dict]:
    """Extract text from screenshot via Hailo-8 OCR"""
    try:
        data = json.dumps({"image_path": image_path}).encode()
        req = urllib.request.Request(f"{HAILO_URL}/vision/ocr", data=data,
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=15)
        return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

def llm_chat(prompt: str, system: str = "", model: str = "gpt-4o") -> str:
    """Send chat to Rust backend or LocalAI"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    try:
        data = json.dumps({
            "model": model, "messages": messages,
            "max_tokens": 2048, "temperature": 0.3
        }).encode()
        req = urllib.request.Request(f"{RUST_BACKEND}/v1/chat/completions",
            data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read().decode())
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"[Error: {e}]"

# ── Browser-Use Controller ───────────────────────────────────

def browser_navigate(url: str, output_dir: str) -> dict:
    """
    Use browser-use to navigate to a URL and capture info.
    Returns: {title, url, screenshot_path, text_content, error}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_path / "screenshot.png"
    
    # Generate browser-use Python script
    script = textwrap.dedent(f'''
    import asyncio, json, os, base64
    os.environ["OLLAMA_HOST"] = "http://192.168.1.47:11434"
    
    from browser_use import Agent, ChatOllama
    
    async def main():
        llm = ChatOllama(model="gemma3:12b", ollama_options={{"num_predict": 128, "temperature": 0.1}})
        agent = Agent(
            task="Go to {url} and extract the page title and visible text content. Take a screenshot.",
            llm=llm,
        )
        history = await agent.run(max_steps=3)
        
        # Take screenshot
        page = agent.browser_session.page if hasattr(agent, 'browser_session') else None
        if page:
            await page.screenshot(path="{screenshot_path}")
        
        result = {{
            "url": "{url}",
            "steps": len(history),
            "final_result": str(history.final_result()) if history else "No result",
            "screenshot": "{screenshot_path}",
        }}
        print(json.dumps(result))
    
    asyncio.run(main())
    ''')
    
    script_path = output_path / "browser_script.py"
    script_path.write_text(script)
    
    # Run via WSL (where browser-use is installed)
    import subprocess
    cmd = f'wsl -d Ubuntu-24.04 -- bash -c "cd /tmp && OLLAMA_HOST=http://192.168.1.47:11434 python3 {script_path}"'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout.strip()
        
        # Try to parse JSON from output
        json_match = re.search(r'\{.*\}', output, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            data["error"] = None
            if not screenshot_path.exists():
                data["screenshot"] = None
            return data
        else:
            return {
                "url": url, "steps": 0,
                "final_result": output[:500],
                "screenshot": str(screenshot_path) if screenshot_path.exists() else None,
                "error": "No JSON in output"
            }
    except subprocess.TimeoutExpired:
        return {"url": url, "error": "Browser timeout (60s)", "screenshot": None}
    except Exception as e:
        return {"url": url, "error": str(e), "screenshot": None}

def analyze_screenshot(screenshot_path: str, url: str) -> dict:
    """
    Full visual analysis of a screenshot using Hailo-8.
    Returns combined results.
    """
    if not screenshot_path or not Path(screenshot_path).exists():
        return {"error": "No screenshot", "url": url}
    
    # Hailo-8 visual analysis
    detections = hailo_detect(screenshot_path)
    ocr_result = hailo_ocr(screenshot_path)
    
    # Extract text from OCR
    page_text = ""
    if ocr_result and "text" in ocr_result:
        page_text = ocr_result["text"]
    elif ocr_result and "result" in ocr_result:
        page_text = ocr_result["result"]
    
    # Extract UI elements from detections
    ui_elements = []
    if detections and "detections" in detections:
        ui_elements = [
            {"label": d.get("label", "?"), "confidence": d.get("confidence", 0),
             "position": f"({d.get('x', 0):.0f},{d.get('y', 0):.0f})"}
            for d in detections["detections"]
        ]
    
    return {
        "url": url,
        "screenshot": screenshot_path,
        "page_text": page_text[:2000],
        "ui_elements": ui_elements[:20],
        "hailo_available": hailo_health(),
        "detections_raw": detections,
        "ocr_raw": ocr_result,
    }

# ── CogniARC Reasoning ───────────────────────────────────────

def cogniarc_think(observation: dict) -> str:
    """
    Use CogniARC reasoning on the visual observation.
    Sends to Rust backend with cognitive system prompt.
    """
    # Build context from visual observation
    context_parts = [f"URL: {observation.get('url', '?')}"]
    
    if observation.get("page_text"):
        context_parts.append(f"\nVisible Text:\n{observation['page_text'][:1000]}")
    
    if observation.get("ui_elements"):
        elements = "\n".join(
            f"  - {e['label']} (conf: {e['confidence']:.0%}) at {e['position']}"
            for e in observation['ui_elements'][:10]
        )
        context_parts.append(f"\nUI Elements:\n{elements}")
    
    if observation.get("hailo_available") == False:
        context_parts.append("\n⚠️ Hailo-8 unavailable — visual analysis limited")
    
    context = "\n".join(context_parts)
    
    # System prompt with CogniARC 6 drives
    system = textwrap.dedent("""You are CogniARC, a cognitive reasoning engine.
    The user has observed a web page through browser-use + Hailo-8 vision.
    Analyze using 6 drives: 1) Curiosity (what's unknown/interesting?)
    2) Pattern Match (similar pages?) 3) Causal (what leads to what?)
    4) Efficiency (what's the key info?) 5) Memory (relevant past knowledge?)
    6) Verify (what could be wrong?).
    
    Return a short analysis with key observations and a recommendation.""")
    
    prompt = f"Analyze this web page observation:\n\n{context}\n\nWhat do you conclude?"
    
    return llm_chat(prompt, system=system)

# ── CLI Commands ─────────────────────────────────────────────

def cmd_analyze(url: str):
    """Full pipeline: navigate → capture → hail → reason"""
    print(f"\n🔍 CogniARC Vision Sensor — Analyzing: {url}\n")
    
    with tempfile.TemporaryDirectory(prefix="cogniarc-vision-") as tmpdir:
        # Step 1: Navigate
        print("  🌐 Browser-Use: navigating...")
        browser_data = browser_navigate(url, tmpdir)
        if browser_data.get("error"):
            print(f"  ❌ Browser error: {browser_data['error']}")
            return
        
        print(f"  ✅ Page loaded ({browser_data.get('steps', '?')} steps)")
        if browser_data.get("final_result"):
            print(f"  📄 {browser_data['final_result'][:200]}")
        
        # Step 2: Visual analysis
        print("\n  👁️  Hailo-8: analyzing screenshot...")
        screenshot = browser_data.get("screenshot")
        observation = analyze_screenshot(screenshot, url)
        
        if observation.get("error"):
            print(f"  ❌ Vision error: {observation['error']}")
        else:
            hailo_status = "✅" if observation.get("hailo_available") else "⚠️"
            print(f"  {hailo_status} Hailo-8: {len(observation.get('ui_elements', []))} UI elements detected")
            if observation.get("page_text"):
                print(f"  📝 OCR text: {observation['page_text'][:150]}...")
            for el in observation.get("ui_elements", [])[:5]:
                print(f"     🎯 {el['label']} at {el['position']}")
        
        # Step 3: CogniARC reasoning
        print("\n  🧠 CogniARC: reasoning...")
        reasoning = cogniarc_think(observation)
        print(f"\n  {'─'*50}")
        print(f"  📋 Analysis:\n")
        for line in reasoning.split("\n"):
            print(f"     {line}")
        print(f"  {'─'*50}")

def cmd_think(query: str, url: Optional[str] = None):
    """Think about a specific question, optionally with a URL"""
    if url:
        cmd_analyze(url)
        print(f"\n  ❓ Your question: {query}\n")
        answer = llm_chat(
            f"Based on the page analysis above, answer: {query}",
            system="Answer concisely based on the visual observation."
        )
        print(f"  💡 {answer}")
    else:
        # Direct CogniARC thinking
        result = llm_chat(query, system="You are CogniARC. Evaluate using 6 drives.")
        print(f"\n  🧠 CogniARC thinks:\n")
        for line in result.split("\n"):
            print(f"     {line}")

def cmd_watch(url: str, interval: int = 60):
    """Watch a page for changes over time"""
    import time
    print(f"\n👁️  Watching {url} every {interval}s\n")
    
    previous_text = ""
    while True:
        try:
            with tempfile.TemporaryDirectory(prefix="cogniarc-watch-") as tmpdir:
                browser_data = browser_navigate(url, tmpdir)
                screenshot = browser_data.get("screenshot")
                observation = analyze_screenshot(screenshot, url)
                current_text = observation.get("page_text", "")
                
                ts = datetime.now().strftime("%H:%M:%S")
                if current_text and current_text != previous_text:
                    print(f"  [{ts}] 🔄 Page changed!")
                    print(f"     {current_text[:200]}")
                    
                    # Analyze change
                    if previous_text:
                        change_analysis = llm_chat(
                            f"Previous page text: {previous_text[:500]}\n\n"
                            f"Current page text: {current_text[:500]}\n\n"
                            f"What changed?",
                            system="Detect and describe page changes concisely."
                        )
                        print(f"     💡 {change_analysis[:200]}")
                    
                    previous_text = current_text
                else:
                    print(f"  [{ts}] ✓ No change")
            
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n  👋 Stopped")
            break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            time.sleep(interval)

# ── Main ─────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CogniARC Vision Sensor — browser-use + Hailo-8 + reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    sub = parser.add_subparsers(dest="command")
    
    p_analyze = sub.add_parser("analyze", help="Full pipeline: navigate → capture → Hailo → reason")
    p_analyze.add_argument("url", help="URL to analyze")
    
    p_think = sub.add_parser("think", help="Think about a question, optionally with a page")
    p_think.add_argument("query", help="Your question")
    p_think.add_argument("--url", "-u", help="URL to load first")
    
    p_watch = sub.add_parser("watch", help="Watch a page for changes")
    p_watch.add_argument("url", help="URL to watch")
    p_watch.add_argument("--interval", "-i", type=int, default=60, help="Check interval (seconds)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == "analyze":
        cmd_analyze(args.url)
    elif args.command == "think":
        cmd_think(args.query, args.url)
    elif args.command == "watch":
        cmd_watch(args.url, args.interval)

if __name__ == "__main__":
    main()
