
import logging
import sys
import time
import uuid

import joblib
import pandas as pd

from flask import Flask, request, jsonify, g

# --------------------------------
# 1. Configure Application Logging
# --------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

# --------------------------------
# 2. Create Flask Application
# --------------------------------

app = Flask(__name__)

# --------------------------------
# 3. Load ML Model
# --------------------------------

MODEL_FILE = "student_result_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)

except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise

# --------------------------------
# 4. Request Logging Middleware
# --------------------------------

@app.before_request
def before_request():
    g.request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    duration_ms = (time.perf_counter() - g.start_time) * 1000

    level = logging.WARNING if response.status_code >= 400 else logging.INFO

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


# --------------------------------
# 5. Health Endpoint
# --------------------------------

@app.route("/health", methods=["GET"])
def health():
    logger.info("event=health_check")
    return jsonify({
        "service": "student-result-prediction",
        "status": "ok"
    })


# --------------------------------
# 6. Prediction Endpoint
# --------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json(silent=True)

        required_fields = [
            "attendance",
            "internal_marks",
            "assignment_marks",
            "previous_score"
        ]

        if not isinstance(data, dict):
            logger.warning("event=invalid_request reason=invalid_json")
            return jsonify({"error": "Invalid JSON request"}), 400

        missing_fields = [
            field for field in required_fields if field not in data
        ]

        if missing_fields:
            logger.warning(
                "event=validation_error missing_fields=%s",
                missing_fields
            )
            return jsonify({
                "error": "Missing required fields",
                "fields": missing_fields
            }), 422

        # Convert inputs to numeric values
        try:
            payload = {
                field: float(data[field])
                for field in required_fields
            }
        except (ValueError, TypeError):
            logger.warning("event=validation_error reason=non_numeric_input")
            return jsonify({
                "error": "All input features must be numeric"
            }), 422

        # Safe logging: feature names only
        logger.info(
            "event=prediction_request feature_names=%s",
            sorted(payload.keys())
        )

        input_df = pd.DataFrame([payload])

        prediction = model.predict(input_df)[0]

        logger.info(
            "event=prediction_success prediction=%s",
            str(prediction)
        )

        return jsonify({
            "prediction": str(prediction),
            "prediction_code": int(prediction)
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------
# 7. Handle Unknown Routes
# --------------------------------

@app.errorhandler(404)
def not_found(error):
    logger.warning("event=route_not_found path=%s", request.path)

    return jsonify({
        "error": "Route not found"
    }), 404


# --------------------------------
# 8. Run Application
# --------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
