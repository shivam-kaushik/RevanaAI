# test_forecast_agent.py hard coded
import os
import sys
import pandas as pd

try:
    from backend.agents.forecast_agent import ForecastAgent
except Exception as e:
    print("Import error: run this from the project ROOT (where 'backend/' exists).")
    print(e)
    sys.exit(1)

CSV_PATH = "./Retail_Transactions_Dataset.csv"   
PERIODS = 3                                      # forecast horizon in months

def main():
    # 1) Load the dataset
    df = pd.read_csv(CSV_PATH)

    # 2) HARD-CODED column mapping based on dataset
    #    Date column: Date
    #    Revenue column: Total_Cost   (optional: Units = Total_Item)
    if "Date" not in df.columns or "Total_Cost" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Total_Cost' in the CSV.")

    # 3) Parse dates; drop bad rows
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # 4) (Optional) Filter to a slice, e.g., a specific city, store type, or season
    # df = df[df["City"] == "Houston"]
    # df = df[df["Season"] == "Winter"]

    # 5) Call your ForecastAgent with explicit mapping
    agent = ForecastAgent()
    result = agent.generate_forecast(
        data=df,
        periods=PERIODS,
        time_col="Date",
        value_cols=["Total_Cost"]  
    )

    # 6) Print a concise summary
    series = result.get("Total_Cost", {})
    print("\n=== Forecast Summary (Total_Cost) ===")
    print("Status:", series.get("status"))
    if series.get("status") == "ok":
        print("Freq:", series.get("freq"))
        print("History points:", series.get("history_points"))
        print("Last actual:", series.get("last_actual"))
        print("Horizon (months):", series.get("forecast_horizon"))
        print("Trend:", series.get("trend"))
        fc = series.get("forecast", [])
        if fc:
            print("First forecast point:", fc[0])
            print("Last forecast point: ", fc[-1])
    else:
        print(series)

if __name__ == "__main__":
    main()
