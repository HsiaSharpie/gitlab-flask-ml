import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


@app.route("/api/v1/model", methods=["POST"])
def predict():
    resp = request.get_json(force=True)
    model = pickle.load(open("model/linear_model.pkl", "rb"))
    prediction = model.predict([resp["payload"]])
    output = prediction[0]
    return jsonify(output)
    return resp


@app.route("/", methods=["GET"])
def home():
    return jsonify({"hello": "world!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)