#!/usr/bin/env python3
"""
Check accuracy of trained sklearn models
Run this locally to validate model performance scores
"""

import pickle
import numpy as np
from sklearn.model_selection import cross_val_score

# Define paths to your trained models
MODELS = {
    'wine': {
        'path': 'models/pkl/wine.pkl',
        'dataset': 'load_wine',
        'task': 'classification'
    },
    'cancer': {
        'path': 'models/pkl/cancer.pkl',
        'dataset': 'load_breast_cancer',
        'task': 'classification'
    },
    'diabetes': {
        'path': 'models/pkl/diabetes.pkl',
        'dataset': 'load_diabetes',
        'task': 'regression'
    },
    'churn': {
        'path': 'models/pkl/churn.pkl',
        'dataset': 'custom',
        'task': 'classification'
    },
    'aqi': {
        'path': 'models/pkl/aqi.pkl',
        'dataset': 'custom',
        'task': 'regression'
    }
}

def load_dataset(dataset_name):
    """Load sklearn dataset"""
    from sklearn import datasets
    
    if dataset_name == 'load_wine':
        return datasets.load_wine(return_X_y=True)
    elif dataset_name == 'load_breast_cancer':
        return datasets.load_breast_cancer(return_X_y=True)
    elif dataset_name == 'load_diabetes':
        return datasets.load_diabetes(return_X_y=True)
    else:
        return None, None

def check_accuracy(model_name):
    """Check model accuracy with cross-validation"""
    config = MODELS.get(model_name)
    if not config:
        print(f"❌ {model_name}: Unknown model")
        return
    
    try:
        # Load model
        with open(config['path'], 'rb') as f:
            model = pickle.load(f)
        
        # Load dataset
        X, y = load_dataset(config['dataset'])
        if X is None:
            print(f"⚠️  {model_name}: Dataset loading not supported for custom data")
            return
        
        # Calculate accuracy
        if config['task'] == 'classification':
            scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            accuracy = scores.mean()
            std = scores.std()
            print(f"✅ {model_name:12} | {accuracy*100:6.1f}% (±{std*100:.1f}%) | {config['task']}")
        else:
            scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            r2 = scores.mean()
            std = scores.std()
            print(f"✅ {model_name:12} | R²={r2:6.3f} (±{std:.3f}) | {config['task']}")
    
    except FileNotFoundError:
        print(f"❌ {model_name:12} | File not found: {config['path']}")
    except Exception as e:
        print(f"❌ {model_name:12} | Error: {str(e)}")

if __name__ == '__main__':
    print("=" * 70)
    print("ModelHub — Accuracy Checker")
    print("=" * 70)
    print()
    
    for model_name in MODELS.keys():
        check_accuracy(model_name)
    
    print()
    print("=" * 70)
    print("Copy these accuracy values to your Upload form!")
    print("=" * 70)
