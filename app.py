from flask import Flask, render_template, request, jsonify
from circle_game import run_game
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json() # retrieve the data sent from JavaScript
    # process the data using Python code
    print(data['value'])
    result = run_game(data['value']['lines'], data['value']['current_intersections'])
    print(result)
    return jsonify(result=result) # return the result to JavaScript
