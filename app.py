"""
Flask backend for VisaPredictAI
API server for visa processing time prediction and analytics
"""
import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
from preprocess import (
    load_data, create_features, preprocess_for_inference, get_similar_applications
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
model = None
feature_cols = None
reference_df = None
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_visa_data.csv')
model_path = os.path.join(os.path.dirname(__file__), 'model', 'visa_model.pkl')
features_path = os.path.join(os.path.dirname(__file__), 'model', 'feature_names.pkl')


def load_model_and_data():
    """Load trained model and reference data"""
    global model, feature_cols, reference_df
    
    print("🔄 Loading model and data...")
    
    try:
        # Load reference data
        reference_df = load_data(data_path)
        print(f"✓ Loaded reference data: {len(reference_df)} records")
        
        # Load model
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✓ Loaded model from {model_path}")
        else:
            print(f"❌ Model not found at {model_path}")
            print("   Please run: python backend/model/train_dummy_model.py")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load feature names
        if os.path.exists(features_path):
            feature_cols = joblib.load(features_path)
            print(f"✓ Loaded {len(feature_cols)} feature names")
        else:
            print(f"⚠ Feature names not found, using defaults")
            # Default feature list
            feature_cols = [
                'month', 'year', 'quarter', 'day_of_year', 'seasonal_index',
                'country_mean_days', 'office_mean_days', 'visa_type_mean_days', 
                'visa_sub_type_mean_days',
                'country_encoded', 'visa_type_encoded', 'visa_sub_type_encoded', 
                'office_encoded',
                'country_visa_interaction', 'office_month_interaction'
            ]
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


@app.route('/api/options', methods=['GET'])
def get_options():
    """
    GET /api/options
    Returns dropdown options from the dataset
    """
    try:
        if reference_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Get unique values
        countries = sorted(reference_df['applicant_country'].unique().tolist())
        visa_types = sorted(reference_df['visa_type'].unique().tolist())
        offices = sorted(reference_df['processing_office'].unique().tolist())
        years = sorted(reference_df['application_date'].dt.year.unique().tolist())
        
        # Build visa_sub_types mapping
        visa_sub_types = {}
        for vtype in visa_types:
            sub_types = sorted(
                reference_df[reference_df['visa_type'] == vtype]['visa_sub_type'].unique().tolist()
            )
            visa_sub_types[vtype] = sub_types
        
        response = {
            'countries': countries,
            'visa_types': visa_types,
            'visa_sub_types': visa_sub_types,
            'processing_offices': offices,
            'years': years
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Predict visa processing time
    
    Request JSON:
    {
        "country": "India",
        "visa_type": "Work",
        "visa_sub_type": "H1B",
        "office": "New York",
        "date": "2024-05-15"
    }
    
    Response JSON:
    {
        "minDays": 35,
        "maxDays": 45,
        "avgDays": 40,
        "confidence": 85,
        "similarApplications": 25,
        "trend": "increasing"
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['country', 'visa_type', 'visa_sub_type', 'office', 'date']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Preprocess input
        features_dict = preprocess_for_inference(data, reference_df)
        
        # Prepare feature vector in correct order
        X_input = np.array([features_dict[col] for col in feature_cols]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        
        # Generate confidence score (based on data variance and similarity)
        similar_count, avg_similar = get_similar_applications(reference_df, data)
        confidence = min(95, 50 + (similar_count / 2))  # Scale from 50-95 based on similarity
        
        # Calculate min/max range (±20% of prediction)
        min_days = max(1, int(prediction * 0.8))
        max_days = int(prediction * 1.2)
        avg_days = int(prediction)
        
        # Determine trend
        current_month = int(data['date'].split('-')[1])
        monthly_data = reference_df[
            (reference_df['applicant_country'] == data['country']) &
            (reference_df['visa_type'] == data['visa_type'])
        ]
        
        if len(monthly_data) > 0:
            current_month_avg = monthly_data[
                monthly_data['application_date'].dt.month == current_month
            ]['processing_days'].mean()
            overall_avg = monthly_data['processing_days'].mean()
            trend = 'increasing' if current_month_avg > overall_avg else 'decreasing'
        else:
            trend = 'stable'
        
        response = {
            'minDays': min_days,
            'maxDays': max_days,
            'avgDays': avg_days,
            'confidence': int(confidence),
            'similarApplications': int(similar_count),
            'trend': trend,
            'rawPrediction': float(prediction)
        }
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/trends', methods=['GET'])
def get_trends():
    """
    GET /api/trends?country=&visa_type=&office=
    Returns aggregated statistics for trend visualization
    
    Response includes:
    - Average processing time by country
    - Average processing time by visa type
    - Monthly trend
    - Office workload trend
    """
    try:
        if reference_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Get filter parameters
        country = request.args.get('country', None)
        visa_type = request.args.get('visa_type', None)
        office = request.args.get('office', None)
        
        # Apply filters
        df_filtered = reference_df.copy()
        
        if country and country != 'All':
            df_filtered = df_filtered[df_filtered['applicant_country'] == country]
        
        if visa_type and visa_type != 'All':
            df_filtered = df_filtered[df_filtered['visa_type'] == visa_type]
        
        if office and office != 'All':
            df_filtered = df_filtered[df_filtered['processing_office'] == office]
        
        # 1. Average by country
        by_country = df_filtered.groupby('applicant_country')['processing_days'].agg(['mean', 'count']).round(2)
        by_country_data = [
            {'country': idx, 'avg_days': row['mean'], 'count': int(row['count'])}
            for idx, row in by_country.iterrows()
        ]
        
        # 2. Average by visa type
        by_visa_type = df_filtered.groupby('visa_type')['processing_days'].agg(['mean', 'count']).round(2)
        by_visa_type_data = [
            {'visa_type': idx, 'avg_days': row['mean'], 'count': int(row['count'])}
            for idx, row in by_visa_type.iterrows()
        ]
        
        # 3. Monthly trend
        df_filtered['application_month'] = pd.to_datetime(df_filtered['application_date']).dt.to_period('M')
        by_month = df_filtered.groupby('application_month')['processing_days'].agg(['mean', 'count']).round(2)
        by_month_data = [
            {'month': str(idx), 'avg_days': row['mean'], 'count': int(row['count'])}
            for idx, row in by_month.iterrows()
        ]
        
        # 4. Office workload trend
        by_office = df_filtered.groupby('processing_office')['processing_days'].agg(['mean', 'count']).round(2)
        by_office_data = [
            {'office': idx, 'avg_days': row['mean'], 'count': int(row['count'])}
            for idx, row in by_office.iterrows()
        ]
        
        # 5. Overall statistics
        overall_stats = {
            'total_applications': len(df_filtered),
            'avg_processing_days': float(df_filtered['processing_days'].mean().round(2)),
            'min_processing_days': int(df_filtered['processing_days'].min()),
            'max_processing_days': int(df_filtered['processing_days'].max()),
            'std_processing_days': float(df_filtered['processing_days'].std().round(2))
        }
        
        response = {
            'by_country': by_country_data,
            'by_visa_type': by_visa_type_data,
            'monthly_trend': by_month_data,
            'by_office': by_office_data,
            'overall_stats': overall_stats
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'data_loaded': reference_df is not None
    }), 200


@app.route('/', methods=['GET'])
def index():
    """Index endpoint"""
    return jsonify({
        'service': 'VisaPredictAI Backend',
        'version': '1.0.0',
        'endpoints': {
            'GET /api/options': 'Get dropdown options',
            'POST /api/predict': 'Predict processing time',
            'GET /api/trends': 'Get trend analytics',
            'GET /health': 'Health check'
        }
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VISAPREDICTAI BACKEND SERVER")
    print("=" * 60)
    
    # Load model and data
    if load_model_and_data():
        print("\n✅ All systems ready!")
        print("\n🚀 Starting Flask server...")
        print("   Server: http://127.0.0.1:5000")
        print("   Frontend: http://127.0.0.1:5000/../frontend/")
        print("\n" + "=" * 60)
        
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n❌ Failed to start server - Model not loaded")
        sys.exit(1)
