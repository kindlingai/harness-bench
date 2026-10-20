"""The openai-chat harness against a tiny streaming server that plays a fixed tool-calling script."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from harness_bench import registry
from harness_bench.harnesses.openai_chat import OpenAIChatHarness
from harness_bench.runner import run_one


def tool_call_turn(name: str, args: dict) -> list[dict]:
    text = json.dumps(args)
    head = {"index": 0, "id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": ""}}
    chunks = [{"choices": [{"delta": {"tool_calls": [head]}}]}]
    for i in range(0, len(text), 7):  # arguments arrive in pieces, as real servers send them
        part = {"index": 0, "function": {"arguments": text[i : i + 7]}}
        chunks.append({"choices": [{"delta": {"tool_calls": [part]}}]})
    chunks.append({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": len(text) // 4 + 1}})
    return chunks


def serve(script: list[list[dict]]):
    requests = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in script[min(len(requests) - 1, len(script) - 1)]:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, requests


def test_tool_loop_solves_release_notes():
    task_cls = registry.tasks()["release-notes"]
    _, entries = task_cls(1)._plan()
    feats = "".join(f"- {s} (#{n})\n" for k, s, n in entries if k == "feat")
    fixes = "".join(f"- {s} (#{n})\n" for k, s, n in entries if k == "fix")
    notes = ("intro " * 70) + f"\n\n## Features\n\n{feats}\n## Fixes\n\n{fixes}"
    srv, requests = serve(
        [
            tool_call_turn("read_file", {"path": "CHANGES.txt"}),
            tool_call_turn("write_file", {"path": "RELEASE_NOTES.md", "content": notes}),
            tool_call_turn("finish", {"message": "done"}),
        ]
    )
    try:
        h = OpenAIChatHarness(base_url=f"http://127.0.0.1:{srv.server_port}/v1")
        r = run_one(task_cls, 1, h, max_turns=5)
    finally:
        srv.shutdown()
    assert r.passed, r.checks
    assert r.stop == "finish" and h.model == "fake-model"
    assert r.throughput["turns"] == 3
    assert r.throughput["by_class"]["prose"]["tokens"] > 0
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert {t["function"]["name"] for t in requests[0]["tools"]} >= {"read_file", "write_file", "finish"}
