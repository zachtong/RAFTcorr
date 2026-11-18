"""Development launcher: Flask backend on port 5000."""

import sys
import os

# Ensure project root is on sys.path so `server` and `raft_dic_gui` are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.app import create_app, socketio


def main():
    app = create_app()
    print("Starting Flask dev server on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, use_reloader=True)


if __name__ == "__main__":
    main()
