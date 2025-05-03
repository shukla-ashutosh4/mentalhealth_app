import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_model(model_path='models/mental_health_model.pkl'):
    """
    Loading the trained model
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_test_data(data_path='data/processed_data.csv', test_size=0.2):
    """
    Loading and preparing test data
    """
    try:
        # Load the processed data
        data = pd.read_csv(data_path)
        
        # Shuffle data and take the last test_size portion for testing
        data = data.sample(frac=1, random_state=42)
        test_data = data.iloc[int(len(data) * (1 - test_size)):]
        
        print(f"Test data loaded with {len(test_data)} samples")
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def prepare_features(test_data):
    """
    Preparing features for model prediction
    """
    # Define features to use for prediction
    features = ['phq_total', 'gad_total', 'age', 'sleep_hours', 'exercise_weekly']
    
    # Handle gender with one-hot encoding
    if 'gender' in test_data.columns:
        gender_dummies = pd.get_dummies(test_data['gender'], prefix='gender')
        X_test = pd.concat([test_data[features], gender_dummies], axis=1)
    else:
        X_test = test_data[features]
    
    # Target variable
    y_test = test_data['mental_health_status']
    
    return X_test, y_test

def test_model_performance(model, X_test, y_test):
    """
    Testing model performance
    """
    if model is None:
        print("No model to test")
        return
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'predictions': list(y_pred),
        'actual': list(y_test)
    }

def simulate_questionnaire():
    """
    Simulating a user taking the PHQ-9 and GAD-7 questionnaires
    """
    print("\n=== Mental Health Assessment Questionnaire ===")
    print("Please rate how often you've been bothered by the following problems over the last 2 weeks.")
    print("0 - Not at all, 1 - Several days, 2 - More than half the days, 3 - Nearly every day\n")
    
    # PHQ-9 questions
    phq_questions = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep, or sleeping too much",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself or that you are a failure",
        "Trouble concentrating on things",
        "Moving or speaking so slowly that other people could have noticed",
        "Thoughts that you would be better off dead or of hurting yourself",
    ]
    
    # GAD-7 questions
    gad_questions = [
        "Feeling nervous, anxious, or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen",
    ]
    
    # Collect PHQ-9 responses
    print("PHQ-9 Questionnaire:")
    phq_responses = []
    for i, question in enumerate(phq_questions, 1):
        while True:
            try:
                response = int(input(f"Q{i}. {question} (0-3): "))
                if 0 <= response <= 3:
                    phq_responses.append(response)
                    break
                else:
                    print("Please enter a number between 0 and 3")
            except ValueError:
                print("Please enter a valid number")
    
    # Collect GAD-7 responses
    print("\nGAD-7 Questionnaire:")
    gad_responses = []
    for i, question in enumerate(gad_questions, 1):
        while True:
            try:
                response = int(input(f"Q{i}. {question} (0-3): "))
                if 0 <= response <= 3:
                    gad_responses.append(response)
                    break
                else:
                    print("Please enter a number between 0 and 3")
            except ValueError:
                print("Please enter a valid number")
    
    # Collect demographic information
    print("\nDemographic Information:")
    age = int(input("Age: "))
    gender = input("Gender (Male/Female/Non-binary): ")
    sleep_hours = float(input("Average sleep hours per night: "))
    exercise_weekly = int(input("Weekly exercise sessions: "))
    
    # Calculate totals
    phq_total = sum(phq_responses)
    gad_total = sum(gad_responses)
    
    # Create user data dictionary
    user_data = {
        'phq_total': phq_total,
        'gad_total': gad_total,
        'age': age,
        'gender': gender,
        'sleep_hours': sleep_hours,
        'exercise_weekly': exercise_weekly
    }
    
    return user_data

def predict_mental_health(model, user_data):
    """
    Make prediction based on user questionnaire responses
    """
    # Convert user data to DataFrame
    user_df = pd.DataFrame([user_data])
    
    # Handle gender with one-hot encoding
    gender_dummies = pd.get_dummies(user_df['gender'], prefix='gender')
    
    # Prepare features
    features = ['phq_total', 'gad_total', 'age', 'sleep_hours', 'exercise_weekly']
    user_features = pd.concat([user_df[features], gender_dummies], axis=1)
    
    # Make prediction
    prediction = model.predict(user_features)[0]
    
    return prediction

def get_recommendations(mental_health_status):
    """
    Provide recommendations based on mental health status
    """
    # Load recommendations from JSON file
    try:
        with open('data/recommendations.json', 'r') as f:
            recommendations = json.load(f)
        
        if mental_health_status in recommendations:
            return recommendations[mental_health_status]
        else:
            # Default recommendations
            return recommendations.get('default', {
                'resources': ['Please consult with a healthcare professional'],
                'self_care': ['Regular exercise', 'Adequate sleep', 'Balanced diet'],
                'professional_help': ['Consider speaking with a mental health professional']
            })
    except FileNotFoundError:
        # Create basic recommendations dictionary if file not found
        basic_recommendations = {
            'Healthy': {
                'resources': ['Mental health information: https://www.nimh.nih.gov/health/topics/index'],
                'self_care': ['Continue regular exercise', 'Maintain sleep schedule', 'Practice mindfulness'],
                'professional_help': ['Regular check-ups with healthcare provider']
            },
            'Mild Depression': {
                'resources': ['Depression information: https://www.nimh.nih.gov/health/topics/depression/index'],
                'self_care': ['Daily physical activity', 'Establish routine', 'Connect with others'],
                'professional_help': ['Consider speaking with a mental health counselor']
            },
            'Moderate Depression': {
                'resources': ['Depression support: https://www.nami.org/Learn-More/Mental-Health-Conditions/Depression'],
                'self_care': ['Regular exercise', 'Sleep hygiene', 'Social connection'],
                'professional_help': ['Schedule appointment with mental health professional']
            },
            'Severe Depression': {
                'resources': ['Crisis support: Text HOME to 741741, National Suicide Prevention Lifeline: 1-800-273-8255'],
                'self_care': ['Follow treatment plan', 'Connect with support system', 'Self-care activities'],
                'professional_help': ['Urgent consultation with mental health professional']
            },
            'Mild Anxiety': {
                'resources': ['Anxiety information: https://www.nimh.nih.gov/health/topics/anxiety-disorders/index'],
                'self_care': ['Deep breathing exercises', 'Regular exercise', 'Limit caffeine'],
                'professional_help': ['Consider speaking with a mental health counselor']
            },
            'Moderate Anxiety': {
                'resources': ['Anxiety support: https://adaa.org/'],
                'self_care': ['Relaxation techniques', 'Physical activity', 'Mindfulness practice'],
                'professional_help': ['Schedule appointment with mental health professional']
            },
            'Severe Anxiety': {
                'resources': ['Crisis support: Text HOME to 741741'],
                'self_care': ['Follow treatment plan', 'Breathing exercises', 'Grounding techniques'],
                'professional_help': ['Urgent consultation with mental health professional']
            },
            'default': {
                'resources': ['Mental health information: https://www.nimh.nih.gov/health/topics/index'],
                'self_care': ['Regular exercise', 'Adequate sleep', 'Balanced diet'],
                'professional_help': ['Consider speaking with a mental health professional']
            }
        }
        
        # Save recommendations for future use
        os.makedirs('data', exist_ok=True)
        with open('data/recommendations.json', 'w') as f:
            json.dump(basic_recommendations, f, indent=4)
        
        if mental_health_status in basic_recommendations:
            return basic_recommendations[mental_health_status]
        else:
            return basic_recommendations['default']

def main():
    """
    Main function to test the model
    """
    print("\n=== Mental Health Assessment Model Testing ===")
    print("1. Test model on existing data")
    print("2. Simulate a questionnaire and get recommendations")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Load and prepare test data
        test_data = load_test_data()
        if test_data is None:
            return
        
        X_test, y_test = prepare_features(test_data)
        
        # Test model performance
        test_model_performance(model, X_test, y_test)
    
    elif choice == '2':
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Simulate questionnaire
        user_data = simulate_questionnaire()
        
        # Make prediction
        prediction = predict_mental_health(model, user_data)
        
        print(f"\nPredicted Mental Health Status: {prediction}")
        
        # Get recommendations
        recommendations = get_recommendations(prediction)
        
        print("\n=== Personalized Recommendations ===")
        print("\nResources:")
        for resource in recommendations.get('resources', []):
            print(f"- {resource}")
        
        print("\nSelf-Care Strategies:")
        for strategy in recommendations.get('self_care', []):
            print(f"- {strategy}")
        
        print("\nProfessional Help:")
        for help_option in recommendations.get('professional_help', []):
            print(f"- {help_option}")
        
        print("\nDisclaimer: This assessment is not a substitute for professional diagnosis. Please consult with a healthcare provider for proper evaluation and treatment.")
    
    elif choice == '3':
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please try again.")
        main()

if __name__ == "__main__":
    main()