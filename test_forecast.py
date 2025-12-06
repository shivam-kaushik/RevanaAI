import requests
import json
import sys

def test_forecast():
    url = "http://localhost:8000/chat"
    payload = {
        "message": "forecast next 6 months",
        "conversation_id": "test_verification"
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Response Status: OK")
            # Check if forecasts are present
            if data.get("data", {}).get("forecasts"):
                print("✅ Forecast data returned!")
                # print(json.dumps(data.get("data", {}).get("forecasts"), indent=2))
            else:
                 print("⚠️ No forecast data found in response.")
                 print(json.dumps(data, indent=2))
        else:
            print(f"❌ Failed: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_forecast()
