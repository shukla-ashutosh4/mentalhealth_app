"""
This module provides personalized recommendations based on mental health assessment results.
Recommendations are sourced from evidence-based practices and reliable health authorities.
"""

# Recommendations based on depression severity
DEPRESSION_RECOMMENDATIONS = {
    "Minimal or no depression": [
        {
            "title": "Maintain Mental Wellness",
            "description": "Continue your healthy habits to maintain good mental health.",
            "techniques": [
                "Regular physical activity (30 minutes daily)",
                "Balanced nutrition",
                "Consistent sleep schedule (7-9 hours)",
                "Social connections",
                "Mindfulness practice (10 minutes daily)"
            ],
            "resources": [
                "Mental Health America: www.mhanational.org",
                "National Alliance on Mental Illness: www.nami.org"
            ]
        }
    ],
    "Mild depression": [
        {
            "title": "Self-Care Strategies",
            "description": "Implement these evidence-based self-care strategies to improve your mood.",
            "techniques": [
                "Structured daily routine",
                "Physical exercise (30 minutes, 3-5 times weekly)",
                "Light therapy (especially during winter months)",
                "Journaling thoughts and feelings",
                "Pleasant activity scheduling"
            ],
            "resources": [
                "Depression and Bipolar Support Alliance: www.dbsalliance.org",
                "Headspace meditation app"
            ]
        },
        {
            "title": "Social Support",
            "description": "Research shows social connections can significantly improve mood.",
            "techniques": [
                "Schedule regular calls/meetings with friends or family",
                "Join community groups based on your interests",
                "Consider peer support groups"
            ],
            "resources": [
                "Mental Health America Support Groups: www.mhanational.org/find-support-groups",
                "7 Cups (online emotional support): www.7cups.com"
            ]
        }
    ],
    "Moderate depression": [
        {
            "title": "Consider Professional Support",
            "description": "For moderate depression, professional help is recommended alongside self-care.",
            "techniques": [
                "Cognitive Behavioral Therapy (CBT) techniques",
                "Behavioral activation (scheduling pleasant activities)",
                "Structured problem-solving",
                "Setting small, achievable daily goals"
            ],
            "resources": [
                "Psychology Today Therapist Finder: www.psychologytoday.com/us/therapists",
                "Online therapy options: BetterHelp, Talkspace",
                "Crisis Text Line: Text HOME to 741741"
            ]
        },
        {
            "title": "Lifestyle Modifications",
            "description": "These evidence-based lifestyle changes can reduce depression symptoms.",
            "techniques": [
                "Mediterranean diet (rich in omega-3s, low in processed foods)",
                "Regular exercise (30 minutes daily, especially aerobic)",
                "Sleep hygiene improvement",
                "Reducing alcohol and caffeine",
                "Mindfulness-Based Cognitive Therapy exercises"
            ],
            "resources": [
                "National Institute of Mental Health: www.nimh.nih.gov/health/topics/depression",
                "Mindfulness-Based Cognitive Therapy resources: www.mbct.com"
            ]
        }
    ],
    "Moderately severe depression": [
        {
            "title": "Seek Professional Help",
            "description": "At this level of depression, professional treatment is strongly recommended.",
            "techniques": [
                "Schedule an appointment with a mental health professional",
                "Consider both therapy and medication options",
                "Create a safety plan for difficult moments",
                "Establish a daily routine with small, manageable tasks"
            ],
            "resources": [
                "SAMHSA Treatment Locator: findtreatment.samhsa.gov or call 1-800-662-4357",
                "National Suicide Prevention Lifeline: 988 or 1-800-273-8255",
                "Crisis Text Line: Text HOME to 741741"
            ]
        }
    ],
    "Severe depression": [
        {
            "title": "Urgent Professional Help",
            "description": "Severe depression requires immediate professional intervention.",
            "techniques": [
                "Contact a mental health professional or primary care provider immediately",
                "Consider emergency services if having thoughts of self-harm",
                "Have someone stay with you for support if possible",
                "Focus on basic self-care (eating, sleeping, taking prescribed medications)"
            ],
            "resources": [
                "Emergency Services: 911",
                "National Suicide Prevention Lifeline: 988 or 1-800-273-8255 (24/7)",
                "Crisis Text Line: Text HOME to 741741",
                "Nearest emergency room or urgent care mental health facility"
            ]
        }
    ]
}

# Recommendations based on anxiety severity
ANXIETY_RECOMMENDATIONS = {
    "Minimal anxiety": [
        {
            "title": "Maintain Emotional Balance",
            "description": "Continue these practices to maintain mental wellness.",
            "techniques": [
                "Regular physical activity",
                "Mindfulness practice (10 minutes daily)",
                "Balanced nutrition and hydration",
                "Adequate sleep (7-9 hours)",
                "Time in nature"
            ],
            "resources": [
                "Anxiety and Depression Association of America: www.adaa.org",
                "Mindfulness apps: Calm, Insight Timer"
            ]
        }
    ],
    "Mild anxiety": [
        {
            "title": "Stress Management Techniques",
            "description": "These evidence-based techniques can help manage mild anxiety symptoms.",
            "techniques": [
                "Deep breathing exercises (4-7-8 technique)",
                "Progressive muscle relaxation",
                "Regular physical exercise",
                "Limiting caffeine and alcohol",
                "Journaling worries and solutions"
            ],
            "resources": [
                "Anxiety Canada: www.anxietycanada.com/resources",
                "Headspace meditation app"
            ]
        },
        {
            "title": "Cognitive Techniques",
            "description": "Simple cognitive techniques can help manage anxious thoughts.",
            "techniques": [
                "Thought challenging (identifying and questioning anxious thoughts)",
                "Worry scheduling (setting aside specific time to address worries)",
                "Fact-checking anxious predictions",
                "Mindfulness-based stress reduction"
            ],
            "resources": [
                "MoodGYM: moodgym.com.au",
                "CBT workbooks: 'Mind Over Mood', 'The Anxiety and Worry Workbook'"
            ]
        }
    ],
    "Moderate anxiety": [
        {
            "title": "Consider Professional Support",
            "description": "For moderate anxiety, professional guidance is recommended alongside self-help.",
            "techniques": [
                "Cognitive Behavioral Therapy techniques",
                "Exposure techniques (gradual exposure to anxiety triggers)",
                "Structured relaxation practice (20 minutes daily)",
                "Breathing retraining",
                "Regular physical exercise (particularly aerobic)"
            ],
            "resources": [
                "Psychology Today Therapist Finder: www.psychologytoday.com/us/therapists",
                "Online therapy options: BetterHelp, Talkspace",
                "Anxiety and Depression Association of America: www.adaa.org/finding-help"
            ]
        },
        {
            "title": "Lifestyle Modifications",
            "description": "These evidence-based lifestyle changes can reduce anxiety symptoms.",
            "techniques": [
                "Regular sleep schedule",
                "Anti-inflammatory diet (reducing processed foods, increasing omega-3s)",
                "Limiting caffeine, alcohol, and sugar",
                "Daily physical activity",
                "Digital detox periods"
            ],
            "resources": [
                "National Institute of Mental Health: www.nimh.nih.gov/health/topics/anxiety-disorders",
                "Sleep Foundation: www.sleepfoundation.org"
            ]
        }
    ],
    "Severe anxiety": [
        {
            "title": "Seek Professional Help",
            "description": "Severe anxiety requires professional treatment.",
            "techniques": [
                "Schedule an appointment with a mental health professional",
                "Consider both therapy and medication options",
                "Create a crisis plan for panic attacks or severe anxiety episodes",
                "Practice grounding techniques for immediate relief",
                "Maintain basic self-care routines"
            ],
            "resources": [
                "SAMHSA Treatment Locator: findtreatment.samhsa.gov or call 1-800-662-4357",
                "Crisis Text Line: Text HOME to 741741",
                "Anxiety and Depression Association of America therapist finder: adaa.org/finding-help"
            ]
        }
    ]
}

# Combined recommendations based on cluster analysis
CLUSTER_RECOMMENDATIONS = {
    0: [  # Low depression, high anxiety cluster
        {
            "title": "Managing Anxiety with Mindfulness",
            "description": "Mindfulness techniques specifically targeted for anxiety management.",
            "techniques": [
                "Present-moment awareness practices",
                "Body scan meditation",
                "Focused breathing exercises",
                "Mindful walking",
                "Self-compassion meditation"
            ],
            "resources": [
                "Mindfulness-Based Stress Reduction (MBSR) programs",
                "UCLA Mindful Awareness Research Center: www.marc.ucla.edu"
            ]
        }
    ],
    1: [  # High depression, low anxiety cluster
        {
            "title": "Behavioral Activation for Depression",
            "description": "Evidence-based approach to combat depression through activity.",
            "techniques": [
                "Activity scheduling and monitoring",
                "Gradual reintroduction of pleasurable activities",
                "Social engagement planning",
                "Exercise programming (starting small)",
                "Reward-based goal setting"
            ],
            "resources": [
                "Depression and Bipolar Support Alliance: www.dbsalliance.org",
                "Behavioral Activation workbooks and apps"
            ]
        }
    ],
    2: [  # High depression, high anxiety cluster
        {
            "title": "Integrated Treatment Approach",
            "description": "Comprehensive approach for co-occurring depression and anxiety.",
            "techniques": [
                "Unified protocol therapy approach",
                "Emotion regulation skills",
                "Distress tolerance techniques",
                "Structured problem-solving",
                "Interpersonal effectiveness skills"
            ],
            "resources": [
                "National Alliance on Mental Illness: www.nami.org",
                "Dialectical Behavior Therapy resources",
                "Mental Health America: www.mhanational.org"
            ]
        }
    ],
    3: [  # Low depression, low anxiety cluster
        {
            "title": "Preventative Mental Wellness",
            "description": "Strategies to maintain and enhance current good mental health.",
            "techniques": [
                "Positive psychology exercises",
                "Gratitude practices",
                "Strength-based activities",
                "Flow state engagement",
                "Meaningful goal setting"
            ],
            "resources": [
                "Authentic Happiness (UPenn Positive Psychology): www.authentichappiness.sas.upenn.edu",
                "VIA Character Strengths Survey: www.viacharacter.org"
            ]
        }
    ]
}

def get_depression_recommendations(severity):
    """Get recommendations based on depression severity.
    
    Args:
        severity (str): Depression severity interpretation
        
    Returns:
        list: List of recommendation dictionaries
    """
    return DEPRESSION_RECOMMENDATIONS.get(severity, DEPRESSION_RECOMMENDATIONS["Minimal or no depression"])

def get_anxiety_recommendations(severity):
    """Get recommendations based on anxiety severity.
    
    Args:
        severity (str): Anxiety severity interpretation
        
    Returns:
        list: List of recommendation dictionaries
    """
    return ANXIETY_RECOMMENDATIONS.get(severity, ANXIETY_RECOMMENDATIONS["Minimal anxiety"])

def get_cluster_recommendations(cluster_id):
    """Get recommendations based on cluster assignment.
    
    Args:
        cluster_id (int): Cluster ID from KMeans
        
    Returns:
        list: List of recommendation dictionaries
    """
    # Ensure cluster_id is within range
    valid_cluster_id = cluster_id % len(CLUSTER_RECOMMENDATIONS)
    return CLUSTER_RECOMMENDATIONS.get(valid_cluster_id, CLUSTER_RECOMMENDATIONS[0])

def get_personalized_recommendations(phq9_score, gad7_score, cluster_id=None):
    """Get personalized recommendations based on assessment scores and optional cluster.
    
    Args:
        phq9_score (tuple): (score, interpretation) for PHQ-9
        gad7_score (tuple): (score, interpretation) for GAD-7
        cluster_id (int, optional): Cluster ID from KMeans
        
    Returns:
        dict: Combined personalized recommendations
    """
    _, phq9_interpretation = phq9_score
    _, gad7_interpretation = gad7_score
    
    # Get recommendations
    depression_recs = get_depression_recommendations(phq9_interpretation)
    anxiety_recs = get_anxiety_recommendations(gad7_interpretation)
    
    # Combine recommendations
    recommendations = {
        "depression": depression_recs,
        "anxiety": anxiety_recs
    }
    
    # Add cluster recommendations if available
    if cluster_id is not None:
        cluster_recs = get_cluster_recommendations(cluster_id)
        recommendations["personalized"] = cluster_recs
    
    return recommendations