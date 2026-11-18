"""Flask application factory with SocketIO and CORS."""

import os
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

socketio = SocketIO()


def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder=None)

    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-secret-key"),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50 MB upload limit
    )
    if test_config:
        app.config.update(test_config)

    CORS(app, resources={r"/api/*": {
        "origins": "*",
        "expose_headers": ["X-Data-Min", "X-Data-Max"],
    }})

    # Register blueprints
    from server.routes.images import images_bp
    from server.routes.models import models_bp
    from server.routes.roi import roi_bp
    from server.routes.processing import processing_bp
    from server.routes.displacement import displacement_bp
    from server.routes.strain import strain_bp
    from server.routes.probes import probes_bp
    from server.routes.export import export_bp
    from server.routes.arrows import arrows_bp
    from server.routes.quality import quality_bp
    from server.routes.tiling import tiling_bp
    from server.routes.principal_dirs import principal_bp
    from server.routes.session_routes import session_bp

    app.register_blueprint(images_bp, url_prefix="/api/images")
    app.register_blueprint(models_bp, url_prefix="/api/models")
    app.register_blueprint(roi_bp, url_prefix="/api/roi")
    app.register_blueprint(processing_bp, url_prefix="/api/processing")
    app.register_blueprint(displacement_bp, url_prefix="/api/displacement")
    app.register_blueprint(strain_bp, url_prefix="/api/strain")
    app.register_blueprint(probes_bp, url_prefix="/api/probes")
    app.register_blueprint(export_bp, url_prefix="/api/export")
    app.register_blueprint(arrows_bp, url_prefix="/api/arrows")
    app.register_blueprint(quality_bp, url_prefix="/api/quality")
    app.register_blueprint(tiling_bp, url_prefix="/api/tiling")
    app.register_blueprint(principal_bp, url_prefix="/api/principal")
    app.register_blueprint(session_bp, url_prefix="/api/session")

    @app.route("/api/health")
    def health():
        return {"status": "ok"}

    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    return app
