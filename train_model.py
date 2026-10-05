import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Student dataset
data = {
    "attendance": [
        90, 85, 60, 75, 95,
        55, 80, 70, 88, 65,
        92, 58, 78, 82, 68,
        96, 72, 62, 87, 76
    ],

    "internal_marks": [
        85, 80, 55, 70, 90,
        50, 75, 65, 82, 60,
        88, 52, 72, 78, 62,
        94, 68, 58, 84, 73
    ],

    "assignment_marks": [
        90, 85, 60, 72, 95,
        55, 78, 68, 88, 62,
        92, 58, 75, 80, 65,
        96, 70, 60, 86, 74
    ],

    "result": [
        1, 1, 0, 1, 1,
        0, 1, 0, 1, 0,
        1, 0, 1, 1, 0,
        1, 1, 0, 1, 1
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save dataset
df.to_csv("student_results.csv", index=False)

# Input features
X = df[
    [
        "attendance",
        "internal_marks",
        "assignment_marks"
    ]
]

# Target
y = df["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)

# Save trained model
joblib.dump(
    model,
    "student_result_model.pkl"
)

# Save metrics
metrics = {
    "accuracy": accuracy,
    "training_records": len(X_train),
    "testing_records": len(X_test)
}

with open("metrics.json", "w") as file:
    json.dump(
        metrics,
        file,
        indent=4
    )

print("Model training completed successfully.")
print("Accuracy:", accuracy)
print("Generated files:")
print("student_result_model.pkl")
print("metrics.json")
print("student_results.csv")
