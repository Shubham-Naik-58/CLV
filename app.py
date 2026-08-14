import pickle
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


# --- 1. Class Definition (Required for Unpickling) ---
class LinearRegressionFromScratch:

    def __init__(
        self,
        learning_rate=0.05,
        batch_size=512,
        max_epochs=50,
        tolerance=1e-5,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None

    def predict(self, X):
        X_scaled = (X - self.mean) / self.std
        predictions = np.dot(X_scaled, self.weights) + self.bias
        return np.maximum(0, predictions)


# --- 2. Load Pickle Model Directly ---
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract objects from loaded dictionary
#model = model_artifact["model"]
#X_mean = model_artifact["mean"]
#X_std = model_artifact["std"]


# --- 3. Flask API Routes ---
@app.route("/")
def home():
    """Renders the frontend HTML page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract & convert values to float
        recency = float(data["recency"])
        frequency = float(data["frequency"])
        tenure = float(data["tenure"])
        total_historical_spend = np.log1p(
            max(0.0, float(data["total_historical_spend"]))
        )
        avg_order_value = np.log1p(max(0.0, float(data["avg_order_value"])))
        total_items_bought = float(data["total_items_bought"])

        # Construct input matrix
        input_features = np.array(
            [[
                recency,
                frequency,
                tenure,
                total_historical_spend,
                avg_order_value,
                total_items_bought,
            ]]
        )

        # Run inference using class method
        prediction = model.predict(input_features)[0]

        return jsonify({"status": "success", "predicted_spend": float(prediction)})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)