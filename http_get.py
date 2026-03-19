from flask import Flask, request, jsonify

app = Flask(__name__)

# Fake database (just a list)
users = [
    {"id": 1, "name": "Prasanna"},
    {"id": 2, "name": "Alex"}
]

@app.route("/users", methods=["GET"])
def get_users():
    return jsonify(users)