#!/usr/bin/env python3
import sys
import os
import uvicorn
import logging

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    
    print("🚀 Starting Agentic Sales Assistant v2.0...")
    
    try:
        # Import and initialize the app
        from backend.app import app
        
        print("✅ FastAPI app imported successfully")
        
        # Test configuration
        from backend.config import Config
        if Config.OPENAI_API_KEY and Config.OPENAI_API_KEY != "your_openai_api_key_here":
            print("✅ OpenAI API key configured")
        else:
            print("⚠ Warning: OPENAI_API_KEY not set or using default value")
            
        if Config.DATABASE_URL:
            print(f"✅ Database URL configured: {Config.DATABASE_URL.split('@')[-1]}")
        else:
            print("⚠ Warning: DATABASE_URL not set")
        
        # Test database connection
        try:
            from backend.utils.database import db_manager
            if db_manager.test_connection():
                print("✅ Database connection successful")
            else:
                print("❌ Database connection failed - check your PostgreSQL connection")
                return
        except Exception as e:
            print(f"❌ Database error: {e}")
            return
        
        # Display startup information
        print("\n📁 Features:")
        print("   • Dynamic CSV file uploads")
        print("   • Automatic PostgreSQL table creation") 
        print("   • Intelligent agent routing")
        print("   • Conversational ChatGPT for non-data queries")
        
        print("\n🤖 Available Agents:")
        print("   - Planner Agent (Intent Detection & Routing)")
        print("   - SQL Agent (Data Retrieval)")
        print("   - Insight Agent (Analysis & Narratives)") 
        print("   - Forecast Agent (Predictions)")
        print("   - Anomaly Agent (Pattern Detection)")
        
        print("\n📊 Usage:")
        print("   1. Upload a CSV file through the web interface")
        print("   2. Ask questions about your data")
        print("   3. System automatically routes to appropriate agents")
        
        print("\n🌐 Server starting on http://localhost:8000")
        print("   Press Ctrl+C to stop the server\n")
        
        # Start the server - FIXED: Use import string format for reload
        uvicorn.run(
            "backend.app:app",  # Use import string format
            host="localhost",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n💡 Check that:")
        print("1. You're in the project root directory")
        print("2. All packages are installed: pip install -r requirements.txt")
        print("3. The backend/ directory exists with all Python files")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()