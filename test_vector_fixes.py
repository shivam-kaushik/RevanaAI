"""Test the complete vector search flow with the fixes"""
from backend.agents.vector_agent import VectorAgent

print("="*80)
print("🧪 TESTING VECTOR SEARCH WITH FIXES")
print("="*80)

agent = VectorAgent()

# Test 1: Best dairy products (no location)
print("\n" + "="*80)
print("Test 1: 'Provide me best dairy products'")
print("="*80)
query1 = "Provide me best dairy products"
result1 = agent.handle_semantic_query(query1)
print(result1)

# Test 2: Best dairy products in Chicago
print("\n" + "="*80)
print("Test 2: 'Provide me best dairy products in Chicago'")
print("="*80)
query2 = "Provide me best dairy products in Chicago"
result2 = agent.handle_semantic_query(query2)
print(result2)

# Test 3: Best products in New York
print("\n" + "="*80)
print("Test 3: 'Show me the best products in New York'")
print("="*80)
query3 = "Show me the best products in New York"
result3 = agent.handle_semantic_query(query3)
print(result3)

print("\n" + "="*80)
print("✅ ALL TESTS COMPLETE")
print("="*80)
