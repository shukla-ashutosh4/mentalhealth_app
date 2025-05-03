"""
This module handles data processing for training the ML models 
and processing user inputs from the questionnaires.
"""

import pandas as pd
import numpy as np
import os
import random
from sklearn.preprocessing import StandardScaler
from .questionnaires import (
    PHQ9_QUESTIONS, PHQ9_OPTIONS, PHQ9_SCORES, PHQ9_INTERPRETATION,
    GAD7_QUESTIONS, GAD7_OPTIONS, GAD7_SCORES, GAD7_INTERPRETATION,
    get_phq9_score, get_gad7_score
)

def generate_synthetic_data(n_samples=1000, seed=42):
    """Generate synthetic data for model training.
    
    Args:
        n_samples (int): Number of samples to generate
        seed (int): Random seed for reproducibility
        
    Returns:
        pandas.DataFrame: DataFrame with synthetic data
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create empty dataframe
    columns = []
    # Add PHQ-9 question columns
    for i, q in enumerate(PHQ9_QUESTIONS):
        columns.append(f"phq9_q{i+1}")
    
    # Add GAD-7 question columns
    for i, q in enumerate(GAD7_QUESTIONS):
        columns.append(f"gad7_q{i+1}")
    
    # Add score and class columns
    columns.extend(["phq9_score", "gad7_score", "phq9_class", "gad7_class", "mental_health_class"])
    
    data = pd.DataFrame(columns=columns)
    
    # Generate data
    for i in range(n_samples):
        row = {}
        
        # Generate PHQ-9 responses
        phq9_responses = []
        for j in range(len(PHQ9_QUESTIONS)):
            # Create weighted random selection to make data more realistic
            if i % 5 == 0:  # 20% severe cases
                weights = [0.1, 0.2, 0.3, 0.4]  # More likely severe
            elif i % 5 == 1:  # 20% moderate cases
                weights = [0.2, 0.3, 0.3, 0.2]  # Moderate
            elif i % 5 == 2:  # 20% mild cases
                weights = [0.3, 0.4, 0.2, 0.1]  # Mild
            else:  # 40% minimal cases
                weights = [0.6, 0.3, 0.1, 0.0]  # Minimal
                
            response_idx = random.choices(range(len(PHQ9_OPTIONS)), weights=weights)[0]
            response = PHQ9_OPTIONS[response_idx]
            phq9_responses.append(response)
            row[f"phq9_q{j+1}"] = response
        
        # Calculate PHQ-9 score and class
        phq9_total, phq9_interp = get_phq9_score(phq9_responses)
        row["phq9_score"] = phq9_total
        row["phq9_class"] = phq9_interp
        
        # Generate GAD-7 responses with correlation to PHQ-9
        gad7_responses = []
        for j in range(len(GAD7_QUESTIONS)):
            # Create weighted random selection with some correlation to PHQ-9
            if phq9_total >= 15:  # High depression often has high anxiety
                weights = [0.1, 0.2, 0.3, 0.4]  # More likely severe
            elif 10 <= phq9_total < 15:
                weights = [0.2, 0.3, 0.3, 0.2]  # Moderate
            elif 5 <= phq9_total < 10:
                weights = [0.3, 0.4, 0.2, 0.1]  # Mild
            else:
                weights = [0.6, 0.3, 0.1, 0.0]  # Minimal
                
            # Add some randomness to break perfect correlation
            weights = [w + random.uniform(-0.1, 0.1) for w in weights]
            weights = [max(0.01, w) for w in weights]  # Ensure no negative weights
            
            response_idx = random.choices(range(len(GAD7_OPTIONS)), weights=weights)[0]
            response = GAD7_OPTIONS[response_idx]
            gad7_responses.append(response)
            row[f"gad7_q{j+1}"] = response
        
        # Calculate GAD-7 score and class
        gad7_total, gad7_interp = get_gad7_score(gad7_responses)
        row["gad7_score"] = gad7_total
        row["gad7_class"] = gad7_interp
        
        # Create overall mental health class (simplified for model training)
        if phq9_total >= 10 and gad7_total >= 10:
            mental_health_class = "Severe"
        elif phq9_total >= 10 or gad7_total >= 10:
            mental_health_class = "Moderate"
        elif phq9_total >= 5 or gad7_total >= 5:
            mental_health_class = "Mild"
        else:
            mental_health_class = "Minimal"
            
        row["mental_health_class"] = mental_health_class
        
        # Add row to dataframe
        data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
    
    return data

def preprocess_data(data):
    """Preprocess data for model training.
    
    Args:
        data (pandas.DataFrame): Raw data
        
    Returns:
        tuple: (X_features, y_labels, X_cluster) for model training
    """
    # Convert categorical responses to numerical scores
    X_features = pd.DataFrame()
    
    # Process PHQ-9 responses
    for i in range(len(PHQ9_QUESTIONS)):
        col = f"phq9_q{i+1}"
        X_features[col] = data[col].map(PHQ9_SCORES)
    
    # Process GAD-7 responses
    for i in range(len(GAD7_QUESTIONS)):
        col = f"gad7_q{i+1}"
        X_features[col] = data[col].map(GAD7_SCORES)
    
    # Extract labels (for classification)
    y_labels = data["mental_health_class"]
    
    # Prepare data for clustering (using scores)
    X_cluster = X_features.copy()
    
    # Scale data for clustering
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)
    
    return X_features, y_labels, X_cluster_scaled, scaler

def save_data(data, output_dir="data", filename="raw_data.csv"):
    """Save data to CSV file.
    
    Args:
        data (pandas.DataFrame): Data to save
        output_dir (str): Directory to save data
        filename (str): Filename for saved data
    """
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save data
    data.to_csv(os.path.join(output_dir, filename), index=False)

def load_data(data_path):
    """Load data from CSV file.
    
    Args:
        data_path (str): Path to CSV file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    return pd.read_csv(data_path)

# utils/data_processor.py

# utils/data_processor.py

from .questionnaires import PHQ9_QUESTIONS, GAD7_QUESTIONS

def process_questionnaire_data(phq_answers, gad_answers):
    """
    phq_answers: list of 9 integers (0–3)
    gad_answers: list of 7 integers (0–3)
    Returns dict with totals and flattened features.
    """
    if len(phq_answers) != len(PHQ9_QUESTIONS) or len(gad_answers) != len(GAD7_QUESTIONS):
        raise ValueError("Answer lists must match question counts")
    
    phq_score = sum(phq_answers)
    gad_score = sum(gad_answers)
    features = phq_answers + gad_answers

    return {
        'phq_score': phq_score,
        'gad_score': gad_score,
        'features': features
    }



def process_user_responses(phq9_responses, gad7_responses, scaler=None):
    """Process user responses for model prediction.
    
    Args:
        phq9_responses (list): List of user responses to PHQ-9 questions
        gad7_responses (list): List of user responses to GAD-7 questions
        scaler (sklearn.preprocessing.StandardScaler, optional): Scaler for clustering
        
    Returns:
        tuple: (features, phq9_score, gad7_score, scaled_features)
    """
    # Convert responses to scores
    phq9_scores = [PHQ9_SCORES[response] for response in phq9_responses]
    gad7_scores = [GAD7_SCORES[response] for response in gad7_responses]
    
    # Create features array
    features = np.array(phq9_scores + gad7_scores).reshape(1, -1)
    
    # Calculate PHQ-9 and GAD-7 scores
    phq9_score = get_phq9_score(phq9_responses)
    gad7_score = get_gad7_score(gad7_responses)
    
    # Scale features for clustering if scaler is provided
    scaled_features = None
    if scaler is not None:
        scaled_features = scaler.transform(features)
    
    return features, phq9_score, gad7_score, scaled_features

def generate_and_save_data(n_samples=1000, output_dir="data"):
    """Generate synthetic data and save it.
    
    Args:
        n_samples (int): Number of samples to generate
        output_dir (str): Directory to save data
    """
    # Generate data
    data = generate_synthetic_data(n_samples)
    
    # Save raw data
    save_data(data, output_dir, "raw_data.csv")
    
    # Preprocess data
    X_features, y_labels, X_cluster_scaled, scaler = preprocess_data(data)
    
    # Create processed data dataframe
    processed_data = X_features.copy()
    processed_data["mental_health_class"] = y_labels
    
    # Save processed data
    save_data(processed_data, output_dir, "processed_data.csv")
    
    return data, X_features, y_labels, X_cluster_scaled, scaler