from flask import request, jsonify
from flask_restful import Resource
from ml.src.utils import infer


class Classify(Resource):

    def get(self):
        return "Welcome to this tutorial"

    def post(self):
        data = infer(request.get_json()['text'])
        return jsonify(data)

