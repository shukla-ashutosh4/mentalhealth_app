"""
MongoDB connection utility for the mental health application.
"""
import os
from pymongo import MongoClient
from datetime import datetime
import uuid

# MongoDB connection string - replace with your MongoDB URI
# You can set this as an environment variable for security
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.environ.get("DB_NAME", "mental_health_app")

class MongoDBConnector:
    """Class to handle MongoDB connections and operations."""
    
    def __init__(self):
        """Initialize MongoDB connection."""
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.responses_collection = self.db["user_responses"]
        self.recommendations_collection = self.db["recommendations"]
        self.profiles_collection = self.db["user_profiles"]
    
    def store_questionnaire_response(self, user_id, questionnaire_name, responses, scores=None):
        """
        Store user questionnaire responses in MongoDB.
        
        Args:
            user_id (str): Unique identifier for the user
            questionnaire_name (str): Name of the questionnaire
            responses (dict): User's responses to questions
            scores (dict, optional): Calculated scores from responses
        
        Returns:
            str: ID of the inserted document
        """
        # Get user profile info if available
        user_profile = self.profiles_collection.find_one({"user_id": user_id})
        
        document = {
            "user_id": user_id,
            "questionnaire_name": questionnaire_name,
            "responses": responses,
            "scores": scores,
            "timestamp": datetime.now(),
            "response_id": str(uuid.uuid4())
        }
        
        # Add user profile data if available
        if user_profile:
            document["user_name"] = user_profile.get("name")
            document["user_age"] = user_profile.get("age")
        
        result = self.responses_collection.insert_one(document)
        return document["response_id"]
    
    def store_recommendation(self, user_id, response_id, recommendations):
        """
        Store recommendations generated for a user.
        
        Args:
            user_id (str): Unique identifier for the user
            response_id (str): ID of the corresponding questionnaire response
            recommendations (list): List of recommendations provided to the user
            
        Returns:
            str: ID of the inserted document
        """
        document = {
            "user_id": user_id,
            "response_id": response_id,
            "recommendations": recommendations,
            "timestamp": datetime.now(),
            "recommendation_id": str(uuid.uuid4())
        }
        
        result = self.recommendations_collection.insert_one(document)
        return document["recommendation_id"]
    
    def store_user_profile(self, user_id, name, age):
        """
        Store or update user profile information.
        
        Args:
            user_id (str): Unique identifier for the user
            name (str): User's name
            age (int): User's age
            
        Returns:
            bool: Success status
        """
        result = self.profiles_collection.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "name": name,
                    "age": age,
                    "last_updated": datetime.now()
                }
            },
            upsert=True
        )
        
        return result.acknowledged
    
    def get_user_profile(self, user_id):
        """
        Retrieve user profile information.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            dict: User profile or None if not found
        """
        profile = self.profiles_collection.find_one(
            {"user_id": user_id},
            {"_id": 0}
        )
        
        return profile
    
    def get_user_history(self, user_id):
        """
        Retrieve history of a user's questionnaire responses and recommendations.
        
        Args:
            user_id (str): Unique identifier for the user
            
        Returns:
            dict: User's response and recommendation history
        """
        # Get user profile
        profile = self.get_user_profile(user_id)
        
        # Get responses
        responses = list(self.responses_collection.find(
            {"user_id": user_id}, 
            {"_id": 0}
        ).sort("timestamp", -1))
        
        # Get recommendations
        recommendations = list(self.recommendations_collection.find(
            {"user_id": user_id}, 
            {"_id": 0}
        ).sort("timestamp", -1))
        
        return {
            "profile": profile,
            "responses": responses,
            "recommendations": recommendations
        }
    
    def close_connection(self):
        """Close the MongoDB connection."""
        self.client.close()

# Create a singleton instance
mongodb = MongoDBConnector()

def get_mongodb():
    """Get the MongoDB connector instance."""
    return mongodb