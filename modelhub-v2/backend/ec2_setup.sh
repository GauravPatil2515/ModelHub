#!/bin/bash
set -e
echo "=== ModelHub EC2 Setup ==="

sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv git -y

mkdir -p ~/modelhub
cd ~/modelhub

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

export S3_BUCKET_NAME="modelhub-models-yourname"
export AWS_DEFAULT_REGION="us-east-1"

pkill gunicorn 2>/dev/null || true

nohup gunicorn \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --timeout 120 \
  --access-logfile ~/modelhub/access.log \
  --error-logfile ~/modelhub/error.log \
  app:app &

sleep 2
curl -s http://localhost:5000/ | python3 -m json.tool
echo "=== Backend running on :5000 ==="
