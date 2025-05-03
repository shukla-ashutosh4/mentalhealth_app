"""
Main Streamlit application file for the Mental Health Assessment & Recommendation System.

This application provides interactive mental health questionnaires (PHQ-9 and GAD-7),
processes user responses through ML models, and delivers personalized recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib
from PIL import Image

# Import custom modules
import sys
sys.path.append(".")
from utils.questionnaires import (
    PHQ9_QUESTIONS, PHQ9_OPTIONS, PHQ9_SCORES, PHQ9_INTERPRETATION,
    GAD7_QUESTIONS, GAD7_OPTIONS, GAD7_SCORES, GAD7_INTERPRETATION,
    get_phq9_score, get_gad7_score, get_combined_features
)
from utils.recommendations import get_personalized_recommendations
from utils.data_processor import process_user_responses
from models.model_trainer import generate_and_train_models, load_models

# Set page configuration
st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_DIR = "models/saved_models"
DATA_DIR = "data"

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    .recommendation-card {
        background-color: #000000;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 5px solid #4CAF50;
    }
    .title-text {
        color: #1E3A8A;
        font-size: 20px;
        font-weight: 600;
    }
    .container {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: #000000;
        border: 1px solid #000000;
    }
    .assessment-title {
        font-size: 24px;
        font-weight: bold;
        color: #2C3E50;
        margin-bottom: 20px;
    }
    .phq9-container {
        border-left: 5px solid #000000;
        padding-left: 15px;
    }
    .gad7-container {
        border-left: 5px solid #000000;
        padding-left: 15px;
    }
    .question-text {
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: ##000000;
        border-radius: 5px;
        padding: 10px;
        border-left: 5px solid #2196F3;
        margin-bottom: 15px;
    }
    .result-container {
        background-color: #000000;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .severe {
        color: #C0392B;
        font-weight: bold;
    }
    .moderate {
        color: #E67E22;
        font-weight: bold;
    }
    .mild {
        color: #F1C40F;
        font-weight: bold;
    }
    .minimal {
        color: #27AE60;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_or_load_models():
    """Create or load ML models."""
    try:
        # Check if models exist
        if os.path.exists(os.path.join(MODEL_DIR, "classifier.pkl")):
            with st.spinner("Loading ML models..."):
                classifier, kmeans, scaler = load_models(MODEL_DIR)
        else:
            # Train models if they don't exist
            with st.spinner("First-time setup: Generating data and training models..."):
                classifier, kmeans, scaler = generate_and_train_models(n_samples=5000, 
                                                                       data_dir=DATA_DIR, 
                                                                       model_dir=MODEL_DIR)
        
        return classifier, kmeans, scaler
    
    except Exception as e:
        st.error(f"Error loading or creating models: {e}")
        return None, None, None

def app_header():
    """Display app header and information."""
    col1, col2 = st.columns([1, 3])
    
    # with col1:
    #     st.image("https://img.icons8.com/color/240/brain--v2.png", width=100)
    
    with col2:
        st.title("🧠 BrainAI Mental Health Assessment")
        st.markdown("""
        This app provides a confidential mental health assessment using validated screening tools 
        and offers personalized recommendations based on your responses.
        """)
    
    st.markdown("""
    <div class="info-box">
    <strong>Note:</strong> This tool is for educational purposes only and does not replace professional medical advice. 
    If you're experiencing a mental health emergency, please call your local emergency number or mental health crisis line immediately.
    </div>
    """, unsafe_allow_html=True)

def intro_page():
    """Display introduction page."""
    st.markdown("""
    ## Welcome to Your Mental Health Assessment

    This assessment consists of two standardized questionnaires:
    
    1. **PHQ-9 (Patient Health Questionnaire-9)** - A 9-question tool to screen for depression
    2. **GAD-7 (Generalized Anxiety Disorder-7)** - A 7-question tool to screen for anxiety
    
    ### How it works:
    
    1. Answer all questions honestly based on your experiences over the past 2 weeks
    2. Our AI system will analyze your responses
    3. You'll receive personalized recommendations based on your results
    
    ### Privacy Information:
    
    * Your responses are not stored permanently
    * This is a confidential assessment
    * No personal identifying information is collected

    *Taking care of your mental health is just as important as your physical health.*
    """)
    
    if st.button("Begin Assessment", type="primary", use_container_width=True):
        st.session_state.page = "assessment"
        st.rerun()

def display_question(question_text, options, key_prefix, question_index):
    """Display a single question with radio button options."""
    st.markdown(f"""
    <p class="question-text">{question_index+1}. {question_text}</p>
    """, unsafe_allow_html=True)
    
    response = st.radio(
        f"*Over the last 2 weeks, how often have you been bothered by:*",
        options,
        key=f"{key_prefix}_{question_index}",
        label_visibility="collapsed"
    )
    
    return response

def assessment_page():
    """Display assessment questionnaires."""
    st.markdown("<div class='assessment-title'>Mental Health Assessment</div>", 
                unsafe_allow_html=True)
    
    # Create tabs for the questionnaires
    phq9_tab, gad7_tab = st.tabs(["Depression Assessment (PHQ-9)", "Anxiety Assessment (GAD-7)"])
    
    # Initialize responses in session state if not present
    if "phq9_responses" not in st.session_state:
        st.session_state.phq9_responses = ["Not at all"] * len(PHQ9_QUESTIONS)
    
    if "gad7_responses" not in st.session_state:
        st.session_state.gad7_responses = ["Not at all"] * len(GAD7_QUESTIONS)
    
    # PHQ-9 Tab
    with phq9_tab:
        st.markdown("""
        <div class="phq9-container">
        <h3>PHQ-9: Depression Screening</h3>
        <p>Over the last 2 weeks, how often have you been bothered by the following problems?</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, question in enumerate(PHQ9_QUESTIONS):
            response = display_question(question, PHQ9_OPTIONS, "phq9", i)
            st.session_state.phq9_responses[i] = response
            
            if i < len(PHQ9_QUESTIONS) - 1:
                st.divider()
    
    # GAD-7 Tab
    with gad7_tab:
        st.markdown("""
        <div class="gad7-container">
        <h3>GAD-7: Anxiety Screening</h3>
        <p>Over the last 2 weeks, how often have you been bothered by the following problems?</p>
        </div>
        """, unsafe_allow_html=True)
        
        for i, question in enumerate(GAD7_QUESTIONS):
            response = display_question(question, GAD7_OPTIONS, "gad7", i)
            st.session_state.gad7_responses[i] = response
            
            if i < len(GAD7_QUESTIONS) - 1:
                st.divider()
    
    # Submit button
    if st.button("Submit Assessment", type="primary", use_container_width=True):
        with st.spinner("Processing your responses..."):
            time.sleep(1)  # Simulate processing
            st.session_state.page = "results"
            st.rerun()

def severity_class(interpretation):
    """Get CSS class based on severity interpretation."""
    if "severe" in interpretation.lower():
        return "severe"
    elif "moderate" in interpretation.lower():
        return "moderate"
    elif "mild" in interpretation.lower():
        return "mild"
    else:
        return "minimal"

def create_gauge_chart(score, max_score, title, color_scale):
    """Create a gauge chart for visualizing scores."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_score], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white"},
            'steps': color_scale,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return fig

def display_results(classifier, kmeans, scaler):
    """Display assessment results and recommendations."""
    st.markdown("<div class='assessment-title'>Your Assessment Results</div>", 
                unsafe_allow_html=True)
    
    # Process responses
    features, phq9_result, gad7_result, scaled_features = process_user_responses(
        st.session_state.phq9_responses,
        st.session_state.gad7_responses,
        scaler
    )
    
    # Get cluster assignment
    cluster_id = kmeans.predict(scaled_features)[0]
    
    # Get model prediction
    mental_health_class = classifier.predict(features)[0]
    
    # Display score cards
    col1, col2 = st.columns(2)
    
    phq9_score, phq9_interpretation = phq9_result
    gad7_score, gad7_interpretation = gad7_result
    
    with col1:
        st.markdown(f"""
        <div class="container phq9-container">
            <h3>Depression Assessment (PHQ-9)</h3>
            <h4>Score: {phq9_score}/27</h4>
            <p>Interpretation: <span class="{severity_class(phq9_interpretation)}">{phq9_interpretation}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # PHQ-9 gauge chart
        phq9_color_scale = [
            {'range': [0, 4], 'color': 'rgba(39, 174, 96, 0.3)'},
            {'range': [5, 9], 'color': 'rgba(241, 196, 15, 0.3)'},
            {'range': [10, 14], 'color': 'rgba(230, 126, 34, 0.3)'},
            {'range': [15, 19], 'color': 'rgba(231, 76, 60, 0.3)'},
            {'range': [20, 27], 'color': 'rgba(192, 57, 43, 0.3)'}
        ]
        st.plotly_chart(create_gauge_chart(phq9_score, 27, "Depression Severity", phq9_color_scale))
    
    with col2:
        st.markdown(f"""
        <div class="container gad7-container">
            <h3>Anxiety Assessment (GAD-7)</h3>
            <h4>Score: {gad7_score}/21</h4>
            <p>Interpretation: <span class="{severity_class(gad7_interpretation)}">{gad7_interpretation}</span></p>
        </div>
        """, unsafe_allow_html=True)
        
        # GAD-7 gauge chart
        gad7_color_scale = [
            {'range': [0, 4], 'color': 'rgba(39, 174, 96, 0.3)'},
            {'range': [5, 9], 'color': 'rgba(241, 196, 15, 0.3)'},
            {'range': [10, 14], 'color': 'rgba(230, 126, 34, 0.3)'},
            {'range': [15, 21], 'color': 'rgba(192, 57, 43, 0.3)'}
        ]
        st.plotly_chart(create_gauge_chart(gad7_score, 21, "Anxiety Severity", gad7_color_scale))
    
    # Display overall assessment
    st.markdown(f"""
    <div class="result-container">
        <h3>Overall Assessment</h3>
        <p>Based on our AI model analysis, your responses indicate: <span class="{severity_class(mental_health_class)}">{mental_health_class} mental health concerns</span></p>
        <p>You've been assigned to Profile Group {cluster_id+1}, which helps us personalize recommendations for you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get personalized recommendations
    recommendations = get_personalized_recommendations(phq9_result, gad7_result, cluster_id)
    
    # Display recommendations
    st.markdown("## Personalized Recommendations")
    
    recommendation_tabs = st.tabs(["Depression Support", "Anxiety Support", "Personalized Plan"])
    
    # Depression recommendations
    with recommendation_tabs[0]:
        for rec in recommendations["depression"]:
            st.markdown(f"""
            <div class="recommendation-card">
                <p class="title-text">{rec['title']}</p>
                <p>{rec['description']}</p>
                <h4>Suggested Techniques:</h4>
                <ul>
                    {"".join([f"<li>{technique}</li>" for technique in rec['techniques']])}
                </ul>
                <h4>Resources:</h4>
                <ul>
                    {"".join([f"<li>{resource}</li>" for resource in rec['resources']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Anxiety recommendations
    with recommendation_tabs[1]:
        for rec in recommendations["anxiety"]:
            st.markdown(f"""
            <div class="recommendation-card">
                <p class="title-text">{rec['title']}</p>
                <p>{rec['description']}</p>
                <h4>Suggested Techniques:</h4>
                <ul>
                    {"".join([f"<li>{technique}</li>" for technique in rec['techniques']])}
                </ul>
                <h4>Resources:</h4>
                <ul>
                    {"".join([f"<li>{resource}</li>" for resource in rec['resources']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Personalized recommendations
    with recommendation_tabs[2]:
        for rec in recommendations.get("personalized", []):
            st.markdown(f"""
            <div class="recommendation-card">
                <p class="title-text">{rec['title']}</p>
                <p>{rec['description']}</p>
                <h4>Suggested Techniques:</h4>
                <ul>
                    {"".join([f"<li>{technique}</li>" for technique in rec['techniques']])}
                </ul>
                <h4>Resources:</h4>
                <ul>
                    {"".join([f"<li>{resource}</li>" for resource in rec['resources']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Final notes and disclaimer
    st.markdown("""
    <div class="info-box">
    <strong>Important:</strong> These recommendations are based on your assessment results and are meant to
    provide general guidance. They are not a substitute for professional mental health care.
    If your assessment indicates moderate to severe concerns, we strongly encourage you to reach
    out to a mental health professional for a comprehensive evaluation.
    </div>
    """, unsafe_allow_html=True)
    
    # Restart button
    if st.button("Start New Assessment", type="primary"):
        st.session_state.clear()
        st.rerun()

def main():
    """Main application function."""
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "intro"
    
    # Create or load models
    classifier, kmeans, scaler = create_or_load_models()
    
    # Display app header
    app_header()
    
    # Display appropriate page based on session state
    if st.session_state.page == "intro":
        intro_page()
    elif st.session_state.page == "assessment":
        assessment_page()
    elif st.session_state.page == "results":
        if classifier is not None and kmeans is not None and scaler is not None:
            display_results(classifier, kmeans, scaler)
        else:
            st.error("There was an error loading the models. Please refresh the page and try again.")
            if st.button("Go back to home", type="primary"):
                st.session_state.page = "intro"
                st.rerun()

if __name__ == "__main__":
    main()



# """
# Mental Health Application with MongoDB integration
# """
# import streamlit as st
# import pandas as pd
# import pickle
# import uuid
# import os
# from datetime import datetime

# # Import local modules
# from utils.load_env import load_environment
# from utils.data_processor import process_questionnaire_data
# from utils.questionnaires import load_questionnaires
# from utils.recommendations import generate_recommendations
# from utils.mongodb_connector import get_mongodb

# # Load ML models
# @st.cache_resource
# def load_models():
#     with open('models/saved_models/classifier.pkl', 'rb') as f:
#         classifier = pickle.load(f)
#     with open('models/saved_models/kmeans.pkl', 'rb') as f:
#         kmeans = pickle.load(f)
#     return classifier, kmeans

# # Initialize session state
# def init_session():
#     if 'user_id' not in st.session_state:
#         # Generate a unique user ID for the session
#         st.session_state.user_id = str(uuid.uuid4())
#     if 'page' not in st.session_state:
#         st.session_state.page = 'welcome'
#     if 'responses' not in st.session_state:
#         st.session_state.responses = {}
#     if 'current_questionnaire' not in st.session_state:
#         st.session_state.current_questionnaire = None
#     if 'history_view' not in st.session_state:
#         st.session_state.history_view = False
#     if 'user_info_submitted' not in st.session_state:
#         st.session_state.user_info_submitted = False
#     if 'user_name' not in st.session_state:
#         st.session_state.user_name = ""
#     if 'user_age' not in st.session_state:
#         st.session_state.user_age = None

# # Main application
# def main():
#     st.title("Mental Health Assessment & Recommendations")
    
#     # Initialize session
#     init_session()
    
#     # Navigation sidebar
#     with st.sidebar:
#         # Show user info if available
#         if st.session_state.user_info_submitted:
#             st.write(f"**Name:** {st.session_state.user_name}")
#             st.write(f"**Age:** {st.session_state.user_age}")
#         else:
#             st.write("Please provide your information")
        
#         st.write(f"Session ID: {st.session_state.user_id[:8]}...")
        
#         if st.button("Home"):
#             st.session_state.page = 'welcome'
#             st.session_state.history_view = False
        
#         if st.button("View History"):
#             st.session_state.page = 'history'
#             st.session_state.history_view = True
            
#         # Only show assessments if user info has been submitted
#         if st.session_state.user_info_submitted:
#             st.divider()
#             st.write("Available Assessments:")
#             questionnaires = load_questionnaires()
            
#             for q_name in questionnaires.keys():
#                 if st.button(q_name):
#                     st.session_state.page = 'questionnaire'
#                     st.session_state.current_questionnaire = q_name
#                     st.session_state.responses = {}
#                     st.session_state.history_view = False
#         else:
#             st.info("Please provide your information on the home page to access assessments.")
    
#     # Page router
#     if st.session_state.page == 'welcome':
#         render_welcome_page()
#     elif st.session_state.page == 'questionnaire':
#         render_questionnaire_page()
#     elif st.session_state.page == 'results':
#         render_results_page()
#     elif st.session_state.page == 'history':
#         render_history_page()

# def render_welcome_page():
#     st.write("""
#     Welcome to the Mental Health Assessment Tool. This application helps you:
    
#     1. Complete scientifically validated mental health questionnaires
#     2. Receive instant analysis of your responses
#     3. Get personalized recommendations based on your results
#     """)
    
#     st.info("All your responses are stored securely in our database. Your data is kept private and confidential.")
    
#     # Ask for user information if not already provided
#     if not st.session_state.user_info_submitted:
#         st.subheader("Please introduce yourself")
        
#         with st.form("user_info_form"):
#             name = st.text_input("Your Name", key="input_name")
#             age = st.number_input("Your Age", min_value=13, max_value=120, step=1, key="input_age")
            
#             submit_button = st.form_submit_button("Save Information")
            
#             if submit_button:
#                 if name and age:
#                     st.session_state.user_name = name
#                     st.session_state.user_age = age
#                     st.session_state.user_info_submitted = True
                    
#                     # Store user info in MongoDB
#                     mongodb = get_mongodb()
#                     mongodb.db.user_profiles.update_one(
#                         {"user_id": st.session_state.user_id},
#                         {
#                             "$set": {
#                                 "name": name,
#                                 "age": age,
#                                 "last_updated": datetime.now()
#                             }
#                         },
#                         upsert=True
#                     )
                    
#                     st.success("Information saved successfully!")
#                     st.experimental_rerun()
#                 else:
#                     st.error("Please fill in all fields.")
#     else:
#         st.write(f"Hello, **{st.session_state.user_name}**! Please select an assessment from the sidebar to begin.")

# def render_questionnaire_page():
#     questionnaires = load_questionnaires()
#     q_name = st.session_state.current_questionnaire
    
#     if q_name not in questionnaires:
#         st.error("Questionnaire not found.")
#         return
    
#     questionnaire = questionnaires[q_name]
    
#     st.subheader(q_name)
#     st.write(questionnaire.get('description', ''))
    
#     # Create a form for the questionnaire
#     with st.form(key=f"form_{q_name}"):
#         responses = {}
        
#         for q_id, question in questionnaire['questions'].items():
#             q_text = question['text']
#             q_type = question.get('type', 'likert')
            
#             if q_type == 'likert':
#                 options = question.get('options', [
#                     "Not at all", "Several days", 
#                     "More than half the days", "Nearly every day"
#                 ])
#                 responses[q_id] = st.radio(
#                     q_text, 
#                     options=options,
#                     key=f"q_{q_id}"
#                 )
#             elif q_type == 'binary':
#                 responses[q_id] = st.radio(
#                     q_text,
#                     options=["No", "Yes"],
#                     key=f"q_{q_id}"
#                 )
#             elif q_type == 'text':
#                 responses[q_id] = st.text_area(
#                     q_text,
#                     key=f"q_{q_id}"
#                 )
        
#         submit_button = st.form_submit_button(label='Submit Responses')
        
#         if submit_button:
#             st.session_state.responses = responses
            
#             # Process the responses
#             processed_data = process_questionnaire_data(
#                 q_name, 
#                 responses, 
#                 questionnaire
#             )
            
#             # Calculate scores
#             scores = questionnaire.get('scoring_function', lambda x: {"total": sum(x.values())})(processed_data)
            
#             # Store in MongoDB
#             mongodb = get_mongodb()
#             response_id = mongodb.store_questionnaire_response(
#                 st.session_state.user_id,
#                 q_name,
#                 responses,
#                 scores
#             )
            
#             st.session_state.current_response_id = response_id
#             st.session_state.current_scores = scores
#             st.session_state.page = 'results'
#             st.experimental_rerun()

# def render_results_page():
#     if not hasattr(st.session_state, 'current_scores'):
#         st.error("No results to display. Please complete a questionnaire first.")
#         return
    
#     st.subheader("Your Assessment Results")
    
#     # Display scores
#     scores = st.session_state.current_scores
#     for score_name, score_value in scores.items():
#         st.metric(label=score_name.replace('_', ' ').title(), value=score_value)
    
#     # Get recommendations based on the questionnaire responses and scores
#     questionnaires = load_questionnaires()
#     q_name = st.session_state.current_questionnaire
    
#     if q_name in questionnaires:
#         # Load ML models for advanced recommendations
#         classifier, kmeans = load_models()
        
#         # Generate recommendations
#         recommendations = generate_recommendations(
#             q_name,
#             st.session_state.responses,
#             scores,
#             classifier,
#             kmeans
#         )
        
#         # Store recommendations in MongoDB
#         mongodb = get_mongodb()
#         mongodb.store_recommendation(
#             st.session_state.user_id,
#             st.session_state.current_response_id,
#             recommendations
#         )
        
#         # Display recommendations
#         st.subheader("Personalized Recommendations")
#         for i, rec in enumerate(recommendations, 1):
#             st.write(f"{i}. {rec}")
    
#     if st.button("Return to Home"):
#         st.session_state.page = 'welcome'
#         st.experimental_rerun()

# def render_history_page():
#     st.subheader("Your Assessment History")
    
#     # Get user history from MongoDB
#     mongodb = get_mongodb()
#     history = mongodb.get_user_history(st.session_state.user_id)
    
#     # Display user profile information
#     if history['profile']:
#         st.write("User Profile:")
#         profile_col1, profile_col2 = st.columns(2)
#         with profile_col1:
#             st.metric("Name", history['profile']['name'])
#         with profile_col2:
#             st.metric("Age", history['profile']['age'])
#         st.divider()
    
#     if not history['responses']:
#         st.info("You haven't completed any assessments yet.")
#         return
    
#     # Display previous assessments
#     st.write("Previous Assessments:")
    
#     for response in history['responses']:
#         with st.expander(f"{response['questionnaire_name']} - {response['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
#             st.write("Scores:")
#             if response['scores']:
#                 for score_name, score_value in response['scores'].items():
#                     st.metric(label=score_name.replace('_', ' ').title(), value=score_value)
            
#             # Find corresponding recommendations
#             matching_recs = next(
#                 (r for r in history['recommendations'] if r['response_id'] == response['response_id']), 
#                 None
#             )
            
#             if matching_recs:
#                 st.write("Recommendations:")
#                 for i, rec in enumerate(matching_recs['recommendations'], 1):
#                     st.write(f"{i}. {rec}")

# if __name__ == "__main__":
#     # Load environment variables
#     env_vars = load_environment()
    
#     # Run the application
#     main()