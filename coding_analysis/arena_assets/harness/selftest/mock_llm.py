#!/usr/bin/env python3
"""Scratch OpenAI-compatible endpoint used only to exercise the copied harness bundle offline.

Scripted per model name: call 1 -> two shell commands, call 2+ -> task_complete.
Answers as a tool call when the request carries `tools`, as plain content otherwise.
Every request body is appended to <log>.
"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = sys.argv[1]
PORT_FILE = sys.argv[2]
calls: dict[str, int] = {}
lock = threading.Lock()

STEP1 = {
    "analysis": "Fresh shell in the task directory.",
    "plan": "Write hello.txt containing the pip index the shell sees, then show it.",
    "commands": [
        {"keystrokes": "echo \"hello index=$PIP_INDEX_URL\" > hello.txt\n", "duration": 0.5},
        {"keystrokes": "cat hello.txt; pwd\n", "duration": 0.5},
    ],
}
DONE = {"analysis": "hello.txt is in place.", "plan": "Nothing left.", "commands": [], "task_complete": True}


def sse(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        model = body.get("model", "?")
        with lock:
            calls[model] = n = calls.get(model, 0) + 1
            with open(LOG, "a") as f:
                f.write(json.dumps({"n": n, "path": self.path, "auth": self.headers.get("Authorization"),
                                    "body": body}) + "\n")
        answer = json.dumps(STEP1 if n == 1 else DONE)
        use_tool = bool(body.get("tools"))
        usage = {"prompt_tokens": 1000 + 100 * n, "completion_tokens": 50, "total_tokens": 1050 + 100 * n}
        base = {"id": f"mock-{n}", "created": int(time.time()), "model": model}
        if not body.get("stream"):
            msg = {"role": "assistant", "content": None if use_tool else answer, "reasoning_content": f"thinking {n}"}
            if use_tool:
                msg["tool_calls"] = [{"id": f"call_{n}", "type": "function",
                                      "function": {"name": body["tools"][0]["function"]["name"], "arguments": answer}}]
            out = json.dumps({**base, "object": "chat.completion", "usage": usage,
                              "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
            return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close"); self.end_headers()
        ch = {**base, "object": "chat.completion.chunk"}
        def delta(d, fin=None):
            self.wfile.write(sse({**ch, "choices": [{"index": 0, "delta": d, "finish_reason": fin}]}))
        delta({"role": "assistant", "reasoning_content": f"thinking {n}"})
        half = len(answer) // 2
        for idx, part in enumerate((answer[:half], answer[half:])):
            if use_tool:
                first = idx == 0
                fn = {"arguments": part}
                tc = {"index": 0, "function": fn}
                if first:
                    tc.update(id=f"call_{n}", type="function"); fn["name"] = body["tools"][0]["function"]["name"]
                delta({"tool_calls": [tc]})
            else:
                delta({"content": part})
        delta({}, "tool_calls" if use_tool else "stop")
        self.wfile.write(sse({**ch, "choices": [], "usage": usage}))
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        self.close_connection = True


srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
open(PORT_FILE, "w").write(str(srv.server_address[1]))
srv.serve_forever()
