import json
import ast

from bson.json_util import dumps
from flask import request, jsonify, Response
from flask_restful import Resource
from ml.src.utils import Inference
from app.settings import shared


def fetch_posts():
    db = shared["db"]
    collection = db.posts
    try:
        records = collection.find({})
        if records.count() > 0:
            # Prepare the response
            records = dumps(records)
            resp = Response(records, status=200, mimetype='application/json')
            return resp
        else:
            # No records are found
            return Response("No records are found", status=404)
    except Exception as e:
        print("Exception: {}".format(e))
        # Error while trying to fetch the resource
        return Response("Error while trying to fetch the resource", status=500)


def write_post(body):
    db = shared["db"]
    collection = db.posts
    try:
        # Create new users
        collection.insert(body)
        return Response("Successfully created the resource", status=201)

    except Exception as e:
        # Error while trying to create the resource
        print("Exception: {}".format(e))
        return Response("Error while trying to create resource")
    pass


inferer = Inference()


class Classify(Resource):
    def get(self):
        return fetch_posts()
        # return "Welcome to this tutorial"

    def post(self):
        incoming = request.get_json()['text']
        data = inferer.infer(incoming)
        body = {
                    "text": incoming,
                    "result": data

                }
        write_post(body)
        return jsonify(data)

