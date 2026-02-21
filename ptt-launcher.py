#!/usr/bin/env python3
"""Minimal always-on HTTP server for starting push-to-talk from the dashboard.

Listens on port 9848. The main service (port 9847) is unreachable when stopped,
so this sidecar provides the start capability.
"""

import http.server
import json
import subprocess
import time
from pathlib import Path

STATUS_FILE = Path.home() / ".local/share/push-to-talk/status"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b'{}'
        try:
            cmd = json.loads(body)
        except json.JSONDecodeError:
            cmd = {}

        if cmd.get('action') == 'start':
            subprocess.Popen(['systemctl', '--user', 'start', 'push-to-talk.service'])
            # Write restart_live signal so config watcher starts the live session
            time.sleep(0.5)  # Let service start first
            try:
                STATUS_FILE.write_text("restart_live")
            except OSError:
                pass
            self._json_response({"ok": True})
        else:
            self._json_response({"ok": False, "error": "Only 'start' action supported"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # silent


if __name__ == '__main__':
    server = http.server.HTTPServer(('127.0.0.1', 9848), Handler)
    print('PTT launcher listening on http://127.0.0.1:9848', flush=True)
    server.serve_forever()
