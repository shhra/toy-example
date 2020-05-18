from flask import request, jsonify
from flask_restful import Resource
from ml.src.utils import Inference


inferer = Inference()


class Classify(Resource):
    def get(self):
        return "Welcome to this tutorial"

    def post(self):
        data = inferer.infer(request.get_json()['text'])
        return jsonify(data)

