from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST', 'GET'])
def index():
    data = request.get_data()
    print(data)
    return jsonify({"content": "python"})


if __name__ == "__main__":
    # db.create_all()
    app.run(host='0.0.0.0', port=8000)
