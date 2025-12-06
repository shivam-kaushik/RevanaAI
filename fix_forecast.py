try:
    with open('forecast_agent_copy.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply fix
    new_content = content.replace('out["ds"] = pd.to_datetime(out["ds"])', 'out["ds"] = pd.to_datetime(out["ds"], utc=True).dt.tz_localize(None)')
    
    with open('backend/agents/forecast_agent.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Successfully patched forecast_agent.py")
except Exception as e:
    print(f"Error: {e}")
