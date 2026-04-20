from flask import Flask, request, jsonify
from flask_cors import CORS
import boto3, joblib, numpy as np, json, tempfile, os, time

app = Flask(__name__)
CORS(app, origins="*")

S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "modelhub-models-gaurav")
s3 = boto3.client("s3", region_name="ap-south-1")
model_cache = {}
METADATA_KEY = "metadata/models.json"
START_TIME = time.time()

# ── helpers ──
def load_model(model_id):
    if model_id not in model_cache:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_file(S3_BUCKET, f"models/{model_id}.pkl", f.name)
            model_cache[model_id] = joblib.load(f.name)
            os.unlink(f.name)
    return model_cache[model_id]

def get_metadata():
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=METADATA_KEY)
        return json.loads(obj["Body"].read())
    except:
        # Fallback: return hardcoded 5 models if metadata file doesn't exist
        return [
            {"id":"cancer","name":"Breast Cancer Detector","type":"classification","author":"Gaurav Patil","description":"Predicts breast cancer malignancy from tissue measurements","features":["mean radius","mean texture","mean perimeter","mean area","mean smoothness","mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension","radius error","texture error","perimeter error","area error","smoothness error","compactness error","concavity error","concave points error","symmetry error","fractal dimension error","worst radius","worst texture","worst perimeter","worst area","worst smoothness","worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"],"acc":"95.6%","tags":["sklearn","gradient-boost","medical"],"s3":"s3://modelhub-models-gaurav/models/cancer.pkl","runs":0,"n_features":30},
            {"id":"wine","name":"Wine Cultivar Classifier","type":"classification","author":"Gaurav Patil","description":"Classifies wine cultivars from chemical composition","features":["alcohol","malic_acid","ash","alkalinity","magnesium","phenols","flavanoids","nonflavanoids","proanthocyanins","color_intensity","hue","od280_od315","proline"],"acc":"100.0%","tags":["sklearn","svm","wine"],"s3":"s3://modelhub-models-gaurav/models/wine.pkl","runs":0,"n_features":13},
            {"id":"diabetes","name":"Diabetes Progression","type":"regression","author":"Gaurav Patil","description":"Predicts diabetes disease progression from health metrics","features":["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],"acc":"R²=0.437","tags":["sklearn","regression","medical"],"s3":"s3://modelhub-models-gaurav/models/diabetes.pkl","runs":0,"n_features":10},
            {"id":"churn","name":"Customer Churn Predictor","type":"classification","author":"Gaurav Patil","description":"Predicts customer churn from contract and usage data","features":["tenure","monthly_charges","total_charges","contract_type","internet_service","payment_method","paperless_billing","phone_service"],"acc":"98.5%","tags":["sklearn","random-forest","business"],"s3":"s3://modelhub-models-gaurav/models/churn.pkl","runs":0,"n_features":5},
            {"id":"aqi","name":"Air Quality Index","type":"regression","author":"Gaurav Patil","description":"Predicts AQI from air pollutant concentrations","features":["co","no2","o3","pm25","temp","humidity"],"acc":"R²=0.918","tags":["sklearn","regression","environment"],"s3":"s3://modelhub-models-gaurav/models/aqi.pkl","runs":0,"n_features":6}
        ]

def save_metadata(data):
    s3.put_object(
        Bucket=S3_BUCKET, Key=METADATA_KEY,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )

# ── routes ──
@app.route("/")
def health():
    uptime_sec = int(time.time() - START_TIME)
    h, r = divmod(uptime_sec, 3600)
    m, s = divmod(r, 60)
    try:
        meta = boto3.client("ec2").describe_instances()
        instance_id = meta["Reservations"][0]["Instances"][0]["InstanceId"]
    except:
        instance_id = "local"
    return jsonify({
        "status": "ok",
        "service": "ModelHub API v2",
        "ec2_instance": instance_id,
        "s3_bucket": S3_BUCKET,
        "region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "uptime": f"{h}h {m}m {s}s",
        "models_cached": len(model_cache)
    })

@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify(get_metadata())

@app.route("/api/validate", methods=["POST"])
def validate_model():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file"}), 400
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        file.save(f.name)
        try:
            model = joblib.load(f.name)
            n_feat = getattr(model, "n_features_in_", None)
            classes = list(map(str, getattr(model, "classes_", [])))
            model_type = type(model).__name__
            has_proba = hasattr(model, "predict_proba")
            # test run
            if n_feat:
                model.predict(np.zeros((1, n_feat)))
            return jsonify({
                "valid": True,
                "model_type": model_type,
                "n_features": n_feat,
                "classes": classes,
                "has_proba": has_proba,
                "message": f"Valid sklearn model: {model_type} — {n_feat} features"
            })
        except Exception as e:
            return jsonify({"valid": False, "error": str(e)}), 400
        finally:
            os.unlink(f.name)

@app.route("/api/upload", methods=["POST"])
def upload_model():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400
    model_id = f"model-{int(time.time())}"
    meta = {
        "id": model_id,
        "name": request.form.get("name", "Unnamed Model"),
        "author": request.form.get("author", "Anonymous"),
        "description": request.form.get("description", ""),
        "type": request.form.get("type", "classification"),
        "accuracy": request.form.get("accuracy", "N/A"),
        "features": [f.strip() for f in request.form.get("features", "feature_1").split(",")],
        "tags": [t.strip() for t in request.form.get("tags", "").split(",") if t.strip()],
        "runs": 0,
        "uploaded_at": int(time.time()),
        "s3_path": f"s3://{S3_BUCKET}/models/{model_id}.pkl"
    }
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        file.save(f.name)
        try:
            model = joblib.load(f.name)
            n_feat = getattr(model, "n_features_in_", len(meta["features"]))
            np.zeros((1, n_feat))  # test shape
            model.predict(np.zeros((1, n_feat)))
            meta["n_features"] = int(n_feat)
            s3.upload_file(f.name, S3_BUCKET, f"models/{model_id}.pkl")
        except Exception as e:
            return jsonify({"error": f"Invalid model: {e}"}), 400
        finally:
            os.unlink(f.name)
    all_models = get_metadata()
    all_models.insert(0, meta)
    save_metadata(all_models)
    return jsonify({"success": True, "model": meta})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json or {}
    model_id = data.get("model")
    inp = data.get("input", {})
    if not model_id:
        return jsonify({"error": "model id required"}), 400
    t0 = time.time()
    try:
        model = load_model(model_id)
        X = np.array(list(inp.values()), dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        result = {
            "prediction": str(pred),
            "source": f"s3://{S3_BUCKET}/models/{model_id}.pkl",
            "latency_ms": round((time.time() - t0) * 1000, 1)
        }
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            result["confidence"] = round(float(max(probs)) * 100, 1)
            result["probabilities"] = {
                str(c): round(float(p) * 100, 1)
                for c, p in zip(model.classes_, probs)
            }
        # increment run counter (best effort, don't crash if fails)
        try:
            all_models = get_metadata()
            for m in all_models:
                if m["id"] == model_id:
                    m["runs"] = m.get("runs", 0) + 1
            save_metadata(all_models)
        except Exception as e:
            print(f"Warning: Could not update metadata: {e}")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
