"""
This module contains the questions and scoring logic for standardized mental health questionnaires.
"""

# PHQ-9 Questionnaire for Depression Screening
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down?",
    "Trouble concentrating on things, such as reading the newspaper or watching television?",
    "Moving or speaking so slowly that other people could have noticed? Or so fidgety or restless that you have been moving a lot more than usual?",
    "Thoughts that you would be better off dead, or thoughts of hurting yourself in some way?"
]

PHQ9_OPTIONS = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day"
]

PHQ9_SCORES = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

PHQ9_INTERPRETATION = {
    (0, 4): "Minimal or no depression",
    (5, 9): "Mild depression",
    (10, 14): "Moderate depression",
    (15, 19): "Moderately severe depression",
    (20, 27): "Severe depression"
}

# GAD-7 Questionnaire for Anxiety Screening
GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge?",
    "Not being able to stop or control worrying?",
    "Worrying too much about different things?",
    "Trouble relaxing?",
    "Being so restless that it's hard to sit still?",
    "Becoming easily annoyed or irritable?",
    "Feeling afraid as if something awful might happen?"
]

GAD7_OPTIONS = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day"
]

GAD7_SCORES = {
    "Not at all": 0,
    "Several days": 1,
    "More than half the days": 2,
    "Nearly every day": 3
}

GAD7_INTERPRETATION = {
    (0, 4): "Minimal anxiety",
    (5, 9): "Mild anxiety",
    (10, 14): "Moderate anxiety",
    (15, 21): "Severe anxiety"
}

def get_phq9_score(responses):
    """Calculate the PHQ-9 score from responses.
    
    Args:
        responses (list): List of responses to PHQ-9 questions
        
    Returns:
        tuple: (total_score, interpretation)
    """
    total_score = sum(PHQ9_SCORES[response] for response in responses)
    
    # Determine interpretation
    interpretation = None
    for score_range, interp in PHQ9_INTERPRETATION.items():
        if score_range[0] <= total_score <= score_range[1]:
            interpretation = interp
            break
    
    return total_score, interpretation

def get_gad7_score(responses):
    """Calculate the GAD-7 score from responses.
    
    Args:
        responses (list): List of responses to GAD-7 questions
        
    Returns:
        tuple: (total_score, interpretation)
    """
    total_score = sum(GAD7_SCORES[response] for response in responses)
    
    # Determine interpretation
    interpretation = None
    for score_range, interp in GAD7_INTERPRETATION.items():
        if score_range[0] <= total_score <= score_range[1]:
            interpretation = interp
            break
    
    return total_score, interpretation

def get_combined_features(phq9_responses, gad7_responses):
    """Combine responses from both questionnaires into a single feature vector.
    
    Args:
        phq9_responses (list): List of responses to PHQ-9 questions
        gad7_responses (list): List of responses to GAD-7 questions
        
    Returns:
        list: Combined feature vector of scores
    """
    phq9_scores = [PHQ9_SCORES[response] for response in phq9_responses]
    gad7_scores = [GAD7_SCORES[response] for response in gad7_responses]
    
    # Combine both sets of scores
    combined_features = phq9_scores + gad7_scores
    
    return combined_features