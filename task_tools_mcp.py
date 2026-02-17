#!/usr/bin/env python3
"""
MCP server for push-to-talk task management tools.

Runs as a subprocess of Claude CLI (stdio transport).
Proxies tool calls to the main push-to-talk process via Unix domain socket.
"""

import sys
import os
import json
import socket

SOCKET_PATH = os.environ.get("PTT_TOOL_SOCKET", "")

# Tool definitions in MCP format
TOOLS = [
    {
        "name": "spawn_task",
        "description": (
            "Start a Claude CLI task in the background. The task has full tool access -- "
            "Bash, file read/write/edit, grep, glob, web search, everything. Use for ANY "
            "request that touches files, code, commands, or the real world. Return immediately "
            "with a brief acknowledgment. Keep acknowledgments short, every word takes time to speak."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short descriptive name, 2-4 words"
                },
                "prompt": {
                    "type": "string",
                    "description": "Detailed prompt for Claude CLI to execute"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Absolute path to the project directory where Claude should work"
                }
            },
            "required": ["name", "prompt", "project_dir"]
        }
    },
    {
        "name": "list_tasks",
        "description": (
            "List all tasks with their current status. Use when the user asks what tasks "
            "are running or wants a status update. Summarize concisely -- every word costs "
            "time to speak aloud."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_task_status",
        "description": "Get status of a specific task by name or number.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                }
            },
            "required": ["identifier"]
        }
    },
    {
        "name": "get_task_result",
        "description": (
            "Read a task's output. For completed tasks, summarizes what was accomplished. "
            "For running tasks, shows recent progress. Keep spoken summaries brief."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                },
                "tail_lines": {
                    "type": "integer",
                    "description": "Number of output lines to return from the end"
                }
            },
            "required": ["identifier"]
        }
    },
    {
        "name": "cancel_task",
        "description": "Cancel a running task. No confirmation needed -- just do it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "description": "Task name, partial name, or ID number"
                }
            },
            "required": ["identifier"]
        }
    },
    {
        "name": "send_notification",
        "description": (
            "Send a desktop notification immediately. Use for reminders, alerts, timers, "
            "or any time the user asks to be notified about something. Executes instantly "
            "-- no need to spawn a task for this."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Notification title (short, a few words)"
                },
                "body": {
                    "type": "string",
                    "description": "Notification body text"
                },
                "urgency": {
                    "type": "string",
                    "enum": ["low", "normal", "critical"],
                    "description": "Urgency level (default: normal)"
                }
            },
            "required": ["title", "body"]
        }
    },
]


def send_response(response):
    """Write a JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def call_main_process(tool_name, arguments):
    """Call the main push-to-talk process via Unix socket to execute a tool."""
    if not SOCKET_PATH:
        return {"error": "PTT_TOOL_SOCKET not set"}

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30)  # 30s timeout for tool execution
        sock.connect(SOCKET_PATH)

        request = json.dumps({"tool": tool_name, "args": arguments})
        sock.sendall(request.encode() + b"\n")

        # Read response (newline-delimited)
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in data:
                break

        return json.loads(data.decode().strip())
    except socket.timeout:
        return {"error": f"Tool '{tool_name}' timed out"}
    except ConnectionRefusedError:
        return {"error": "Main process not available"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        sock.close()


def main():
    """Main MCP server loop — read JSON-RPC from stdin, write responses to stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = request.get("method", "")
        req_id = request.get("id")

        # JSON-RPC notifications (no id) don't get responses
        if req_id is None and method.startswith("notifications/"):
            continue

        if method == "initialize":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "ptt-task-tools", "version": "1.0.0"}
                }
            })

        elif method == "tools/list":
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOLS}
            })

        elif method == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                result = call_main_process(tool_name, arguments)
                result_text = json.dumps(result) if isinstance(result, dict) else str(result)
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": result_text}]
                    }
                })
            except Exception as e:
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                        "isError": True
                    }
                })

        elif req_id is not None:
            # Unknown method with an id — return error
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            })


if __name__ == "__main__":
    main()
