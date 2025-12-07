"""Test script to verify visualization intent is properly detected"""
import sys
sys.path.insert(0, 'c:\\Users\\Public\\Documents\\My_Projects\\Revana')

from backend.utils.intent_detector import IntentDetector

# Initialize detector
detector = IntentDetector()

# Test queries
test_queries = [
    "Provide me pie chart for sales per category",
    "Show me a bar chart of revenue by month",
    "Create a line graph for sales trends",
    "Visualize customer purchases",
    "Give me a pie chart",
    "Show sales data with a chart"
]

print("=" * 60)
print("Testing Visualization Intent Classification")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    result = detector.detect_intent(query, has_active_dataset=True)
    
    print(f"   ✓ Data Query: {result['is_data_query']}")
    print(f"   ✓ Primary Intent: {result['primary_intent']}")
    print(f"   ✓ Required Agents: {result['required_agents']}")
    
    # Check if VISUALIZATION_AGENT and INSIGHT_AGENT are included
    has_viz = "VISUALIZATION_AGENT" in result['required_agents']
    has_sql = "SQL_AGENT" in result['required_agents']
    has_insight = "INSIGHT_AGENT" in result['required_agents']
    
    if has_viz and has_sql and has_insight:
        print("   ✅ ALL CORRECT AGENTS INCLUDED (SQL + VISUALIZATION + INSIGHT)")
    else:
        print(f"   ❌ MISSING AGENTS - SQL: {has_sql}, VIZ: {has_viz}, INSIGHT: {has_insight}")

print("\n" + "=" * 60)
