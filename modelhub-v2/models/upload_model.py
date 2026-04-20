import subprocess, sys, os, json, argparse

EC2_IP  = "3.109.62.81"          # ← your EC2 IP
KEY     = "/home/gaurav/Desktop/modelhub-key.pem"  # ← your SSH key path
BUCKET  = "modelhub-models-gaurav"
REGION  = "ap-south-1"

parser = argparse.ArgumentParser()
parser.add_argument('pkl',    help='Path to .pkl file')
parser.add_argument('--name', required=True, help='Model display name')
parser.add_argument('--type', default='classification',
                    choices=['classification','regression','nlp','clustering'])
parser.add_argument('--acc',      default='N/A')
parser.add_argument('--features', default='feature_1',
                    help='Comma separated feature names')
parser.add_argument('--tags',     default='sklearn')
parser.add_argument('--desc',     default='')
args = parser.parse_args()

pkl_path = os.path.abspath(args.pkl)
filename = os.path.basename(pkl_path)
model_id = filename.replace('.pkl', '')

if not os.path.exists(pkl_path):
    print(f"❌ File not found: {pkl_path}")
    sys.exit(1)

print(f"📦 Uploading {filename} to EC2...")
scp = subprocess.run([
    'scp', '-i', KEY,
    '-o', 'StrictHostKeyChecking=no',
    pkl_path,
    f'ubuntu@{EC2_IP}:~/modelhub/{filename}'
], capture_output=True, text=True)

if scp.returncode != 0:
    print(f"❌ SCP failed: {scp.stderr}")
    sys.exit(1)
print(f"✅ File uploaded to EC2")

# Register on EC2 via SSH
register_cmd = f"""
cd ~/modelhub && source venv/bin/activate && python3 - << 'PYEOF'
import boto3, json, pickle
import numpy as np

s3     = boto3.client('s3', region_name='{REGION}')
BUCKET = '{BUCKET}'

s3.upload_file('/home/ubuntu/modelhub/{filename}', BUCKET, 'models/{filename}')

try:
    obj    = s3.get_object(Bucket=BUCKET, Key='metadata/models.json')
    models = json.loads(obj['Body'].read())
except:
    models = []

existing_ids = [m['id'] for m in models]
if '{model_id}' not in existing_ids:
    models.insert(0, {{
        "id":          "{model_id}",
        "name":        "{args.name}",
        "type":        "{args.type}",
        "description": "{args.desc or args.name + ' — uploaded from VS Code'}",
        "author":      "Gaurav Patil",
        "accuracy":    "{args.acc}",
        "features":    {json.dumps([f.strip() for f in args.features.split(',')])},
        "tags":        {json.dumps([t.strip() for t in args.tags.split(',') if t.strip()])},
        "s3_path":     f"s3://{BUCKET}/models/{filename}",
        "runs":        0
    }})
    s3.put_object(Bucket=BUCKET, Key='metadata/models.json',
        Body=json.dumps(models, indent=2), ContentType='application/json')
    print(f"registered total={{len(models)}}")
else:
    print("already_exists")
PYEOF
"""

ssh = subprocess.run([
    'ssh', '-i', KEY,
    '-o', 'StrictHostKeyChecking=no',
    f'ubuntu@{EC2_IP}', register_cmd
], capture_output=True, text=True)

if 'registered' in ssh.stdout:
    total = ssh.stdout.strip().split('=')[-1]
    print(f"✅ Registered in S3 metadata — Total models: {total}")
    print(f"🌐 View at: http://{EC2_IP}:5000/api/models")
elif 'already_exists' in ssh.stdout:
    print(f"⚠️  Model '{model_id}' already exists in registry")
else:
    print(f"❌ SSH Error: {ssh.stderr}")
