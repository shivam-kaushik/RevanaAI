#!/usr/bin/env python3
import sys
import os

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import and run the app
from backend.app import app

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Agentic Sales Assistant...")
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000, 
        reload=True,
        log_level="info"
    )