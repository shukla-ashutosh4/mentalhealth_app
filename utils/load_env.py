"""
Load environment variables for the application.
"""
import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if required environment variables are set
    required_vars = ["MONGO_URI", "DB_NAME"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or set them manually.")
        
    return {var: os.environ.get(var) for var in required_vars}