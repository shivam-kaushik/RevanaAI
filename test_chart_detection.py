"""Test script to verify pie chart generation"""
import sys
sys.path.insert(0, 'c:\\Users\\Public\\Documents\\My_Projects\\Revana')

from backend.agents.data_analyzer import DataAnalyzer

# Initialize analyzer
analyzer = DataAnalyzer()

# Test queries and sample data
test_data = [
    {'product_category': 'Electronics', 'total_sales': 156905},
    {'product_category': 'Clothing', 'total_sales': 155580},
    {'product_category': 'Beauty', 'total_sales': 143515}
]

test_queries = [
    "Provide me pie chart for sales per category",
    "Show me a bar chart of revenue by category",
    "Create a pie chart for product distribution"
]

print("=" * 60)
print("Testing Chart Type Detection")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    
    should_pie = analyzer.should_generate_pie_chart(query, test_data)
    should_bar = analyzer.should_generate_bar_chart(query, test_data)
    should_trend = analyzer.should_generate_trend_chart(query, test_data)
    
    print(f"   Pie Chart: {'✅ YES' if should_pie else '❌ NO'}")
    print(f"   Bar Chart: {'✅ YES' if should_bar else '❌ NO'}")
    print(f"   Trend Chart: {'✅ YES' if should_trend else '❌ NO'}")
    
    # Test actual chart generation
    charts = analyzer.generate_charts(query, test_data)
    if charts:
        print(f"   ✅ Generated charts: {list(charts.keys())}")
    else:
        print(f"   ❌ No charts generated")

print("\n" + "=" * 60)
