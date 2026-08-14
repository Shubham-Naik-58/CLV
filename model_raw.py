import pickle
import numpy as np
import pandas as pd


# --- 1. Custom Linear Regression Class (From Scratch) ---
class LinearRegressionFromScratch:

    def __init__(self, learning_rate=0.05, batch_size=512, max_epochs=50,tolerance=1e-5,):
        self.learning_rate = learning_rate  # Step size for weight updates
        self.batch_size = batch_size  # Number of rows processed per gradient step
        self.max_epochs = max_epochs  # Max passes through the full dataset
        self.tolerance = (
            tolerance  # Minimum loss improvement needed to keep training
        )

        self.weights = None  # Stores learned feature coefficients
        self.bias = 0.0  # Stores learned intercept value
        self.mean = None  # Stores mean of each feature for scaling
        self.std = None  # Stores std dev of each feature for scaling

    def fit(self, X, y):
        n_samples, n_features = X.shape  # Get total row count and column count
        y_true = y.ravel()  # Flatten target variable into a 1D array

        # 1. Feature Normalization
        self.mean = np.mean(X, axis=0)  # Calculate mean across each column
        self.std = np.std(X, axis=0)  # Calculate standard deviation across each column
        self.std[self.std == 0] = 1e-8  # Prevent division by zero for flat features (self.std == 0)Checks
        X_scaled = (X - self.mean) / self.std  # Apply Z-score standardization

        # 2. Parameter Initialization
        self.weights = np.zeros(n_features)  # Initialize all feature weights to 0
        self.bias = 0.0  # Initialize intercept to 0
        previous_loss = float("inf")  # Track loss to measure improvement

        # 3. Epoch Loop (Full passes over data)
        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)  # Shuffle row indices
            X_shuffled = X_scaled[indices]  # Apply shuffle to feature matrix
            y_shuffled = y_true[indices]  # Apply same shuffle to target array

            # Mini-Batch Loop (Step-by-step updates)
            for start_idx in range(0, n_samples, self.batch_size):#Start End Step
                end_idx = start_idx + self.batch_size  # Find batch boundary
                X_batch = X_shuffled[start_idx:end_idx]  # Slice feature mini-batch
                y_batch = y_shuffled[start_idx:end_idx]  # Slice target mini-batch
                b_size = len(X_batch)  # Get current batch row count

                # Forward pass & Error calculation
                preds = (np.dot(X_batch, self.weights) + self.bias)  # Compute batch predictions
                errors = preds - y_batch  # Calculate prediction differences

                # Gradient calculations
                dw = (1 / b_size) * np.dot(X_batch.T, errors)  # Calculate weight derivatives
                db = (1 / b_size) * np.sum(errors)  # Calculate bias derivative

                # Parameter updates
                self.weights -= (self.learning_rate * dw)  # Move weights opposite gradient
                self.bias -= (self.learning_rate * db)  # Move bias opposite gradient

            # 4. Convergence & Loss Check
            epoch_preds = (np.dot(X_scaled, self.weights) + self.bias)  # Full dataset predictions
            current_loss = (np.mean((epoch_preds - y_true) ** 2) / 2)  # Calculate Mean Squared Error loss

            if (abs(previous_loss - current_loss) < self.tolerance):  # Check convergence
                print(f"Converged early at epoch {epoch + 1}!")  # Print stop info
                break  # Exit loop if improvement is trivial
            previous_loss = current_loss  # Save loss for next epoch comparison

        return self  # Return trained model instance

    def predict(self, X):
        X_scaled = (X - self.mean) / self.std  # Scale new data using fit statistics
        predictions = (np.dot(X_scaled, self.weights) + self.bias)  # Multiply features by weights & add bias
        return np.maximum(0, predictions)  # Clip negative values to zero


# --- 2. Feature Aggregation Function ---
def prepare_clv_features(df, cutoff_date, target_months=12):
    """Aggregates transactional logs into customer features and future 12-month target."""
    cutoff_date = pd.to_datetime(cutoff_date)
    target_end_date = cutoff_date + pd.DateOffset(months=target_months)

    # Clean missing CustomerIDs and ensure correct data types
    df = df.dropna(subset=["CustomerID"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # Split transactions into observation window and target window
    obs_df = df[df["InvoiceDate"] <= cutoff_date]
    target_df = df[(df["InvoiceDate"] > cutoff_date) & (df["InvoiceDate"] <= target_end_date)]

    # Extract Historical Customer Features (X)
    features = (
        obs_df.groupby("CustomerID").agg(
            recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days), #lambda -> one liner functins, x = values
            frequency=("InvoiceNo", "nunique"),#(Columns, what to do on them)
            tenure=("InvoiceDate", lambda x: (cutoff_date - x.min()).days),#(cutoff_date - x.min()) gets timedelts object (Days time), . days gets only days
            total_historical_spend=("Revenue", "sum"),
            avg_order_value=("Revenue", "mean"),
            total_items_bought=("Quantity", "sum"),
        ).reset_index()
    )

    # Calculate Actual Target Spend in next 12 months (y)
    target = (
        target_df.groupby("CustomerID")["Revenue"]
        .sum()
        .reset_index()
        .rename(columns={"Revenue": "target_future_spend"})
    )

    # Merge features and targets
    dataset = pd.merge(features, target, on="CustomerID", how="left").fillna(
        {"target_future_spend": 0}
    )

    return dataset


# --- 3. Full Training & Export Pipeline ---
def main():
    # 1. Load raw transactional dataset (Update file name/path if needed)
    print("Loading raw transactional dataset...")
    df = pd.read_excel("Online_Retail.xlsx")

    # 2. Extract features and target variable using a cutoff date
    # Adjust 'cutoff_date' to a midpoint date in your dataset timeline
    cutoff_date = "2011-06-01"
    print(f"Aggregating customer features with cutoff date: {cutoff_date}...")
    dataset = prepare_clv_features(df, cutoff_date=cutoff_date, target_months=12)               #Hello There !!

    # 3. Define feature columns in exact order expected by the model
    feature_cols = [
        "recency",
        "frequency",
        "tenure",
        "total_historical_spend",
        "avg_order_value",
        "total_items_bought",
    ]

    # 4. Log transform skewed money columns
    X_raw = dataset[feature_cols].copy()
    X_raw["total_historical_spend"] = np.log1p(np.maximum(0, X_raw["total_historical_spend"]))
    X_raw["avg_order_value"] = np.log1p(np.maximum(0, X_raw["avg_order_value"]))

    X = X_raw.values
    y = dataset["target_future_spend"].values

    # 5. Standard Scaling (Z-score normalization)
    #X_mean = np.mean(X, axis=0)
    #X_std = np.std(X, axis=0) + 1e-8
    #X_scaled = (X - X_mean) / X_std

    # 6. Train the Linear Regression Model
    print("Training model from scratch...")
    model = LinearRegressionFromScratch()
    model.fit(X, y)

    # 7. Package parameters into dictionary artifact
    #model_artifact = {
        #"model": model,
        #"mean": X_mean,
        #"std": X_std,
        #"feature_cols": feature_cols,
    #}

    # Save trained model instance directly
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model successfully trained on dataset and exported to 'model.pkl'!")


if __name__ == "__main__":
    main()