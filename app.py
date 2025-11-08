from flask import Flask, render_template, request, jsonify
import csv
import time
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Serves a webpage for typing

@app.route('/record', methods=['POST'])
def record_keystrokes():
    data = request.json  # Keystroke data from frontend
    keystrokes = data.get("keystrokes", [])

    if not keystrokes:
        return jsonify({"status": "error", "message": "No keystrokes received"}), 400
    i=1
    while(os.path.exists(f"./data/sample{i}.csv")):
        i+=1
    with open(f"./data/sample{i}.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keystrokes[0].keys())
        writer.writeheader()
        writer.writerows(keystrokes)

    return jsonify({"status": "success", "message": "Keystrokes saved."})

if __name__ == "__main__":
    app.run(debug=True)
