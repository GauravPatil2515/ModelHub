# ModelHub — Mini Hugging Face on AWS

A cloud computing lab project: upload scikit-learn models, auto-deploy to AWS S3, run inference from the browser.

## Stack
- **Frontend**: Single HTML file (Vanilla JS, no framework)
- **Backend**: Flask + Gunicorn on AWS EC2 (t2.micro free tier)
- **Storage**: AWS S3 (model .pkl files + metadata JSON)
- **Auth**: IAM Role on EC2 (no hardcoded credentials)

## Quick Start

### 1. Run Frontend (no server needed)
Open `modelhub-v2.html` directly in your browser.

### 2. Run Backend Locally
```bash
cd backend
python -m venv venv
source venv/bin/activate        # Mac/Linux
pip install -r requirements.txt
export S3_BUCKET_NAME=your-bucket-name
python app.py
# → http://localhost:5000
```

### 3. Deploy to EC2
```bash
# On EC2 after SSH
git clone https://github.com/YOUR_USERNAME/modelhub.git
cd modelhub/backend
chmod +x ec2_setup.sh
./ec2_setup.sh
```

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Health check + AWS info |
| GET | `/api/models` | List all model cards |
| POST | `/api/validate` | Validate a .pkl file |
| POST | `/api/upload` | Upload model to S3 + create card |
| POST | `/api/predict` | Run inference on a model |

## Project Structure
```
modelhub/
├── modelhub-v2.html      # Full frontend (4 pages)
├── README.md
├── .gitignore
└── backend/
    ├── app.py            # Flask API
    ├── requirements.txt
    ├── ec2_setup.sh      # EC2 deploy script
    └── .env.example
```

## AWS Architecture
```
Browser → EC2 (Flask API) → S3 (model files + metadata)
                ↑
           IAM Role (S3 read/write)
```

## Team
Made for Cloud Computing Lab — AI/ML Engineering

**Author**: Gaurav Patil (GauravPatil2515)  
**Last Updated**: April 17, 2026
