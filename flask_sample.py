from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Home Page"

@app.route('/test')
def test():
    return "Test Page"

app.run(debug=True)