import os, json, pickle, time, traceback
import numpy as np
import boto3
import sklearn
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
        # Fallback hardcoded models with full metadata
        return [
            {
                'id': 'cancer',
                'name': 'Cancer Classifier',
                'type': 'classification',
                'desc': 'Predicts malignant/benign tumors using 30 breast cancer features.',
                'author': 'Gaurav Patil',
                'features': ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
                            'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
                            'radius_error', 'texture_error', 'perimeter_error', 'area_error', 'smoothness_error',
                            'compactness_error', 'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
                            'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
                            'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension'],
                'acc': '95.6%',
                'tags': ['sklearn', 'gradient-boost'],
                's3_path': f's3://{BUCKET}/models/cancer.pkl',
                'runs': 143
            },
            {
                'id': 'wine',
                'name': 'Wine Quality Classifier',
                'type': 'classification',
                'desc': 'Classifies wine quality from chemical properties using SVM.',
                'author': 'Gaurav Patil',
                'features': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
                            'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280_od315_of_diluted_wines', 'proline'],
                'acc': '100%',
                'tags': ['sklearn', 'svm'],
                's3_path': f's3://{BUCKET}/models/wine.pkl',
                'runs': 87
            },
            {
                'id': 'diabetes',
                'name': 'Diabetes Progression Predictor',
                'type': 'regression',
                'desc': 'Predicts diabetes disease progression from 10 medical features.',
                'author': 'Gaurav Patil',
                'features': ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'],
                'acc': 'R²=0.437',
                'tags': ['sklearn', 'gradient-boost'],
                's3_path': f's3://{BUCKET}/models/diabetes.pkl',
                'runs': 52
            },
            {
                'id': 'churn',
                'name': 'Customer Churn Predictor',
                'type': 'classification',
                'desc': 'Predicts customer churn from tenure, charges, and support calls.',
                'author': 'Gaurav Patil',
                'features': ['tenure', 'monthly_charges', 'total_charges', 'num_products', 'support_calls'],
                'acc': '98.5%',
                'tags': ['sklearn', 'random-forest'],
                's3_path': f's3://{BUCKET}/models/churn.pkl',
                'runs': 156
            },
            {
                'id': 'aqi',
                'name': 'Air Quality Predictor',
                'type': 'regression',
                'desc': 'Predicts AQI (Air Quality Index) from pollution and weather data.',
                'author': 'Gaurav Patil',
                'features': ['co', 'no2', 'o3', 'pm25', 'temp', 'humidity'],
                'acc': 'R²=0.918',
                'tags': ['sklearn', 'random-forest'],
                's3_path': f's3://{BUCKET}/models/aqi.pkl',
                'runs': 201
            }
        ]

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
        
        data  = f.read()
        model = pickle.loads(data)
        
        # Detect task type
        task = 'regression'
        if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
            task = 'classification'
        
        # Get feature info
        n_features    = int(getattr(model, 'n_features_in_', 0))
        feature_names = list(getattr(model, 'feature_names_in_', 
                        [f'feature_{i}' for i in range(n_features)]))
        
        # Get classes
        classes = []
        if hasattr(model, 'classes_'):
            classes = [str(c) for c in model.classes_]
        
        # Get params (top 8 serializable ones)
        params = {}
        if hasattr(model, 'get_params'):
            try:
                all_params = model.get_params()
                params = {k: str(v) for k, v in list(all_params.items())[:8] 
                          if isinstance(v, (int, float, str, bool, type(None)))}
            except:
                pass
        
        # Get actual class name (handle Pipeline)
        model_class = type(model).__name__
        if model_class == 'Pipeline':
            try:
                last_step = model.steps[-1][1]
                model_class = f'Pipeline → {type(last_step).__name__}'
            except:
                pass
        
        print(f"[validate] ✅ {model_class} ({task}) - {n_features} features, {len(classes)} classes")
        
        return jsonify({
            'valid':           True,
            'model_class':     model_class,
            'task':            task,
            'n_features':      n_features,
            'feature_names':   feature_names,
            'classes':         classes,
            'n_classes':       len(classes),
            'params':          params,
            'file_size_kb':    round(len(data)/1024, 1),
            'sklearn_version': sklearn.__version__
        })
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[validate] ❌ {tb}")
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
