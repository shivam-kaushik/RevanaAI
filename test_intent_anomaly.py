"""Test script to verify anomaly detection intent is properly detected"""
import sys
sys.path.insert(0, 'c:\\Users\\Public\\Documents\\My_Projects\\Revana')

from backend.utils.intent_detector import IntentDetector

# Initialize detector
detector = IntentDetector()

# Test queries
test_queries = [
    "Are there any anomalies in sales?",
    "Show me anomalies",
    "Find outliers in the data",
    "Any unusual patterns?",
    "Are there any spikes in revenue?",
    "Detect anomalies in transactions"
]

print("=" * 60)
print("Testing Anomaly Detection Intent Classification")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    result = detector.detect_intent(query, has_active_dataset=True)
    
    print(f"   ✓ Data Query: {result['is_data_query']}")
    print(f"   ✓ Primary Intent: {result['primary_intent']}")
    print(f"   ✓ Required Agents: {result['required_agents']}")
    print(f"   ✓ Reasoning: {result['reasoning']}")
    
    # Check if ANOMALY_AGENT is included
    if "ANOMALY_AGENT" in result['required_agents']:
        print("   ✅ ANOMALY_AGENT is included - CORRECT!")
    else:
        print("   ❌ ANOMALY_AGENT is missing - INCORRECT!")

print("\n" + "=" * 60)
