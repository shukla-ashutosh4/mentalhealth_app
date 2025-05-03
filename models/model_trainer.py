"""
This module handles training and saving the ML models for mental health assessment.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.data_processor import generate_and_save_data, preprocess_data, load_data

def train_classifier(X, y, model_dir="models/saved_models"):
    """Train a classification model to predict mental health status.
    
    Args:
        X (numpy.ndarray): Features
        y (numpy.ndarray): Labels
        model_dir (str): Directory to save model
        
    Returns:
        sklearn.ensemble.RandomForestClassifier: Trained classifier
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create and train base classifier
    base_clf = RandomForestClassifier(random_state=42)
    
    # Use grid search for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy'
    )
    
    # Fit grid search model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_clf = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save plot
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"))
    
    # Save feature importances
    feature_names = [f"phq9_q{i+1}" for i in range(9)] + [f"gad7_q{i+1}" for i in range(7)]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "feature_importance.png"))
    
    return best_clf

def train_clustering(X, n_clusters=4, model_dir="models/saved_models"):
    """Train a clustering model to identify patterns in responses.
    
    Args:
        X (numpy.ndarray): Features
        n_clusters (int): Number of clusters
        model_dir (str): Directory to save model
        
    Returns:
        sklearn.cluster.KMeans: Trained clustering model
    """
    # Create and train KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Get cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Plot clusters (PCA for visualization if high-dimensional)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.scatter(pca.transform(cluster_centers)[:, 0], pca.transform(cluster_centers)[:, 1], 
                marker='x', s=200, linewidths=3, color='red')
    plt.colorbar(scatter, label='Cluster')
    plt.title('KMeans Clustering Results (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save plot
    plt.savefig(os.path.join(model_dir, "clustering_results.png"))
    
    # Calculate and plot the elbow curve
    distortions = []
    K_range = range(1, 10)
    for k in K_range:
        kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_model.fit(X)
        distortions.append(kmeans_model.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method for Optimal k')
    plt.savefig(os.path.join(model_dir, "elbow_curve.png"))
    
    return kmeans

def save_models(classifier, kmeans, scaler, model_dir="models/saved_models"):
    """Save trained models to disk.
    
    Args:
        classifier (sklearn.ensemble.RandomForestClassifier): Trained classifier
        kmeans (sklearn.cluster.KMeans): Trained clustering model
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler
        model_dir (str): Directory to save models
    """
    # Create directory if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save models
    joblib.dump(classifier, os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(kmeans, os.path.join(model_dir, "kmeans.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    
    print(f"Models saved to {model_dir}")

def load_models(model_dir="models/saved_models"):
    """Load trained models from disk.
    
    Args:
        model_dir (str): Directory where models are saved
        
    Returns:
        tuple: (classifier, kmeans, scaler)
    """
    # Load models
    classifier = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    kmeans = joblib.load(os.path.join(model_dir, "kmeans.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    return classifier, kmeans, scaler

def generate_and_train_models(n_samples=1000, data_dir="data", model_dir="models/saved_models"):
    """Generate data and train models end-to-end.
    
    Args:
        n_samples (int): Number of samples to generate
        data_dir (str): Directory to save data
        model_dir (str): Directory to save models
        
    Returns:
        tuple: (classifier, kmeans, scaler)
    """
    # Step 1: Generate and save data
    print("Generating synthetic data...")
    data, X_features, y_labels, X_cluster_scaled, scaler = generate_and_save_data(n_samples, data_dir)
    
    # Step 2: Train classifier
    print("\nTraining classification model...")
    classifier = train_classifier(X_features, y_labels, model_dir)
    
    # Step 3: Train clustering model
    print("\nTraining clustering model...")
    kmeans = train_clustering(X_cluster_scaled, n_clusters=4, model_dir=model_dir)
    
    # Step 4: Save models
    print("\nSaving models...")
    save_models(classifier, kmeans, scaler, model_dir)
    
    return classifier, kmeans, scaler

if __name__ == "__main__":
    import pandas as pd
    
    # Generate data and train models
    generate_and_train_models(n_samples=5000)