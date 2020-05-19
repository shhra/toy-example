import os

from . import serve
from app.settings import shared
from flask import Flask, Blueprint
from flask_cors import CORS
from flask_restful import Api
from flask_pymongo import PyMongo


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    app.config.from_json('config/config.json')
    mongo = PyMongo(app)
    db = mongo.db
    shared["db"] = db
    print(shared)
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)
    api.add_resource(serve.Classify, '/classify')
    app.register_blueprint(api_bp)
    CORS(app)
    return app
