try:
    with open('backend/agents/forecast_agent.py', 'r', encoding='utf-8') as f:
        print(f.read())
except Exception as e:
    print(f"Error: {e}")
