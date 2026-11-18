"""Production launcher: Flask serves built React frontend + API."""

import atexit
import sys
import os
import argparse
import webbrowser
import threading
import socket
import subprocess

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import create_app, socketio
from flask import send_from_directory


def _kill_stale_server(host, port):
    """Detect and kill any previous server instance occupying our port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        sock.close()
    except (ConnectionRefusedError, OSError, socket.timeout):
        return  # Port is free

    print(f"Port {port} is already in use — killing stale process...")

    if sys.platform == "win32":
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        killed = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                if pid not in killed:
                    subprocess.run(
                        ["taskkill", "/PID", pid, "/F"],
                        capture_output=True,
                    )
                    killed.add(pid)
                    print(f"  Killed PID {pid}")
    else:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], capture_output=True, text=True
        )
        for pid in result.stdout.strip().splitlines():
            if pid.strip():
                subprocess.run(["kill", "-9", pid.strip()], capture_output=True)
                print(f"  Killed PID {pid.strip()}")

    import time
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="RAFTcorr production server")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1, use 0.0.0.0 for Colab)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port number (default: 5000)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window on startup")
    args = parser.parse_args()

    # Kill any stale server on our port before starting
    _kill_stale_server(args.host, args.port)

    app = create_app()

    # Serve the built React frontend
    frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")

    if not os.path.isdir(frontend_dist):
        print(f"ERROR: Frontend build not found at {frontend_dist}")
        print("Run 'cd frontend && npm run build' first.")
        sys.exit(1)

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        # Never let the catch-all intercept API routes — return 404 JSON instead
        if path.startswith("api/"):
            return {"error": "Not found"}, 404
        # Serve static files if they exist, otherwise serve index.html (SPA fallback)
        full_path = os.path.join(frontend_dist, path)
        if path and os.path.isfile(full_path):
            return send_from_directory(frontend_dist, path)
        return send_from_directory(frontend_dist, "index.html")

    host = args.host
    port = args.port
    url = f"http://{host}:{port}"

    if not args.no_browser:
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    # Clean up temp files on exit
    from server.session import session
    atexit.register(session._cleanup_temp_files)

    print(f"Starting RAFTcorr production server at {url}")
    socketio.run(app, host=host, port=port, debug=False, use_reloader=False,
                 allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
