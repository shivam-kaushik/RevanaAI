import shutil
try:
    shutil.copy2('backend/agents/forecast_agent.py', 'forecast_agent_copy.txt')
    print("Success")
except Exception as e:
    print(f"Error: {e}")
