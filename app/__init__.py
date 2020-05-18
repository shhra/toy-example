import os

from flask import Flask, Blueprint
from flask_cors import CORS
from flask_restful import Api


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import serve
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)
    api.add_resource(serve.Classify, '/classify')
    app.register_blueprint(api_bp)
    CORS(app)
    return app
