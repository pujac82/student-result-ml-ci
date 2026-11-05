
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "student_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "student_api_request_duration_seconds",
    "API request duration"
)

PREDICTION_COUNT = Counter(
    "student_predictions_total",
    "Total student predictions",
    ["result"]
)

MODEL_STATUS = Gauge(
    "student_model_status",
    "Model status: 1 = loaded, 0 = not loaded"
)

ERROR_COUNT = Counter(
    "student_api_errors_total",
    "Total API errors",
    ["error_type"]
)


def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        MODEL_STATUS.set(1)
        logger.info("Model loaded successfully")
        return model
    except Exception:
        MODEL_STATUS.set(0)
        logger.exception("Model loading failed")
        return None


model = load_model()


@app.before_request
def before_request():
    g.request_id = str(uuid.uuid4())
    g.start_time = time.time()


@app.after_request
def after_request(response):
    duration = time.time() - g.start_time

    if request.endpoint != "metrics":
        REQUEST_COUNT.labels(
            request.method,
            request.endpoint or "unknown",
            str(response.status_code)
        ).inc()

        REQUEST_DURATION.observe(duration)

    response.headers["X-Request-ID"] = g.request_id

    logger.info(
        "request_id=%s method=%s path=%s status=%s duration=%.4f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration
    )

    return response


@app.route("/")
def home():
    return jsonify({
        "message": "Student Result Prediction API"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


@app.route("/metrics")
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


@app.route("/predict", methods=["POST"])
def predict():

    global model

    if model is None:
        ERROR_COUNT.labels("model_unavailable").inc()
        return jsonify({"error": "Model not available"}), 503

    try:
        data = request.get_json(silent=True) or {}

        missing = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing:
            ERROR_COUNT.labels("missing_fields").inc()
            return jsonify({
                "error": "Missing fields",
                "fields": missing
            }), 400

        values = []

        for feature in FEATURES:
            values.append(float(data[feature]))

        input_data = pd.DataFrame(
            [values],
            columns=FEATURES
        )

        prediction = model.predict(input_data)[0]

        result = "PASS" if int(prediction) == 1 else "FAIL"

        PREDICTION_COUNT.labels(result).inc()

        logger.info(
            "request_id=%s prediction=%s",
            g.request_id,
            result
        )

        return jsonify({
            "prediction": result,
            "prediction_code": int(prediction)
        })

    except (ValueError, TypeError):
        ERROR_COUNT.labels("invalid_input").inc()
        return jsonify({
            "error": "Invalid input values"
        }), 400

    except Exception:
        ERROR_COUNT.labels("prediction_error").inc()
        logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed"
        }), 500


@app.errorhandler(404)
def not_found(error):
    ERROR_COUNT.labels("not_found").inc()
    return jsonify({
        "error": "Endpoint not found"
    }), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
