import os, json, pickle, time, traceback
import numpy as np
import boto3
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET","POST","OPTIONS"])

# Config from env vars
BUCKET   = os.environ.get('S3_BUCKET_NAME', 'modelhub-models-gaurav')
REGION   = os.environ.get('AWS_DEFAULT_REGION', 'ap-south-1')
START_T  = time.time()

def get_s3():
    return boto3.client('s3', region_name=REGION)

def load_model(model_id):
    try:
        s3  = get_s3()
        key = f'models/{model_id}.pkl'
        print(f"[S3] Loading s3://{BUCKET}/{key}")
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        model = pickle.loads(obj['Body'].read())
        print(f"[S3] ✅ Loaded {model_id}")
        return model
    except Exception as e:
        print(f"[S3] ❌ Failed {model_id}: {e}")
        return None

def get_metadata():
    try:
        s3  = get_s3()
        obj = s3.get_object(Bucket=BUCKET, Key='metadata/models.json')
        data = json.loads(obj['Body'].read())
        print(f"[S3] ✅ metadata: {len(data)} models")
        return data
    except Exception as e:
        print(f"[S3] ❌ metadata error: {e}")
        return []

# ── Routes ──────────────────────────────────────────

@app.route('/', strict_slashes=False)
def health():
    uptime_s = int(time.time() - START_T)
    return jsonify({
        'status': 'ok',
        'service': 'ModelHub API v2',
        's3_bucket': BUCKET,
        'region': REGION,
        'uptime': f"{uptime_s//3600}h {(uptime_s%3600)//60}m {uptime_s%60}s"
    })

@app.route('/api/models', strict_slashes=False)
def list_models():
    try:
        models = get_metadata()
        return jsonify({'models': models, 'total': len(models)})
    except Exception as e:
        return jsonify({'models': [], 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST','OPTIONS'], strict_slashes=False)
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        body     = request.get_json(force=True)
        model_id = body.get('model_id','').strip()
        features = body.get('features', [])

        print(f"[predict] model={model_id} features={len(features)}")

        if not model_id:
            return jsonify({'error': 'model_id is required'}), 400
        if not features:
            return jsonify({'error': 'features array is required'}), 400

        model = load_model(model_id)
        if model is None:
            return jsonify({'error': f'Could not load model: {model_id}'}), 500

        X   = np.array(features, dtype=float).reshape(1, -1)
        t0  = time.time()
        pred = model.predict(X)[0]
        lat  = round((time.time() - t0) * 1000, 2)

        result = {
            'prediction':  float(pred),
            'latency_ms':  lat,
            'model_id':    model_id,
            'source':      f's3://{BUCKET}/models/{model_id}.pkl'
        }

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            result['probabilities'] = [round(float(p),4) for p in proba]
            result['confidence']    = round(float(max(proba))*100, 2)

        print(f"[predict] ✅ {model_id} → {pred} ({lat}ms)")
        return jsonify(result)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[predict] ❌ ERROR:\n{tb}")
        return jsonify({'error': str(e), 'trace': tb}), 500

@app.route('/api/validate', methods=['POST','OPTIONS'], strict_slashes=False)
def validate_model():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        f = request.files.get('file')
        if not f:
            return jsonify({'error': 'No file uploaded'}), 400
        model = pickle.loads(f.read())
        n_features = getattr(model, 'n_features_in_', 'unknown')
        return jsonify({
            'valid': True,
            'model_type': type(model).__name__,
            'n_features': n_features
        })
    except Exception as e:
        return jsonify({'valid': False, 'error': str(e)}), 400

@app.route('/api/upload', methods=['POST','OPTIONS'], strict_slashes=False)
def upload_model():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        f           = request.files.get('file')
        name        = request.form.get('name','').strip()
        model_type  = request.form.get('type','classification')
        description = request.form.get('description','')
        author      = request.form.get('author','Gaurav Patil')
        accuracy    = request.form.get('accuracy','N/A')
        tags        = request.form.get('tags','').split(',')

        if not f or not name:
            return jsonify({'error': 'file and name required'}), 400

        model_id = name.lower().replace(' ','_')
        data     = f.read()

        # Validate pkl
        model      = pickle.loads(data)
        n_features = getattr(model, 'n_features_in_', 0)

        # Upload to S3
        s3 = get_s3()
        s3.put_object(Bucket=BUCKET, Key=f'models/{model_id}.pkl', Body=data)

        # Update metadata
        models = get_metadata()
        entry  = {
            'id': model_id, 'name': name, 'type': model_type,
            'description': description, 'author': author,
            'accuracy': accuracy, 'tags': [t.strip() for t in tags if t.strip()],
            'features': [f'feature_{i}' for i in range(n_features)],
            'runs': 0
        }
        models = [m for m in models if m['id'] != model_id]
        models.append(entry)
        s3.put_object(
            Bucket=BUCKET, Key='metadata/models.json',
            Body=json.dumps(models), ContentType='application/json'
        )
        return jsonify({'success': True, 'model_id': model_id, 'n_features': n_features})

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[upload] ❌ {tb}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
