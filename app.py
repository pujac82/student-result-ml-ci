
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd

from flask import Flask, jsonify, request, g

# ----------------------------------
# 1. Configure Application Logging
# ----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

# ----------------------------------
# 2. Create Flask Application
# ----------------------------------

app = Flask(__name__)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]

# ----------------------------------
# 3. Load Model Function
# ----------------------------------

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model file not found. Run train_model.py first."
            )

        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )

        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name
        )
        raise


# ----------------------------------
# 4. Request Logging Middleware
# ----------------------------------

@app.before_request
def before_request():
    g.request_id = (
        request.headers.get("X-Request-ID") or str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):

    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = g.request_id

    return response


# ----------------------------------
# 5. Health Endpoint
# ----------------------------------

@app.get("/")
@app.get("/health")
def health_check():

    logger.info("event=health_check")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    })


# ----------------------------------
# 6. Prediction Endpoint
# ----------------------------------

@app.post("/predict")
def predict():

    try:
        data = request.get_json(silent=True)

        if not data or not isinstance(data, dict):

            logger.warning(
                "event=validation_error reason=invalid_json request_id=%s",
                g.request_id
            )

            return jsonify({
                "error": "JSON request body is required"
            }), 400

        missing = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing:

            logger.warning(
                "event=validation_error missing_fields=%s request_id=%s",
                missing,
                g.request_id
            )

            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        try:
            sample = pd.DataFrame([{
                feature: float(data[feature])
                for feature in FEATURES
            }])

        except (ValueError, TypeError):

            logger.warning(
                "event=validation_error reason=non_numeric_input request_id=%s",
                g.request_id
            )

            return jsonify({
                "error": "All input features must be numeric"
            }), 400

        # Log feature names only; never log full student details
        logger.info(
            "event=prediction_request feature_names=%s request_id=%s",
            FEATURES,
            g.request_id
        )

        # Preserve original model loading and prediction logic
        model = load_model()

        code = int(model.predict(sample)[0])

        result = "PASS" if code == 1 else "FAIL"

        logger.info(
            "event=prediction_success prediction=%s request_id=%s",
            result,
            g.request_id
        )

        return jsonify({
            "prediction": result,
            "prediction_code": code
        })

    except Exception:

        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# ----------------------------------
# 7. Missing Route Error Tracking
# ----------------------------------

@app.errorhandler(404)
def not_found(error):

    logger.warning(
        "event=route_not_found path=%s request_id=%s",
        request.path,
        getattr(g, "request_id", "unknown")
    )

    return jsonify({
        "error": "Route not found"
    }), 404


# ----------------------------------
# 8. Run Application
# ----------------------------------

if __name__ == "__main__":

    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000
    )
