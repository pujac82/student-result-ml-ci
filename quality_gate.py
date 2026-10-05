import json

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]

print("Model accuracy:", accuracy)

MIN_ACCURACY = 0.70

if accuracy >= MIN_ACCURACY:
    print("QUALITY GATE PASSED")
else:
    print("QUALITY GATE FAILED")
    raise SystemExit(1)
