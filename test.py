from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/calculate-route', methods=['POST'])  # Make sure it's POST
def calculate_route():
    data = request.json  # Get JSON data from request
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    start = data.get("start")
    end = data.get("end")

    # Dummy response without Gurobi
    response = {
        "message": "Route calculation successful",
        "start": start,
        "end": end,
        "dummy_path": ["A", "B", "C"]
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # Allows external access
