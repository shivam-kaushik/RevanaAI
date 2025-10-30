import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- Import ForecastAgent ---
try:
    from backend.agents.forecast_agent import ForecastAgent
except Exception as e:
    print("Import error: run this from the project ROOT (where 'backend/' exists).")
    print(e)
    sys.exit(1)

# --- Config ---
CSV_PATH = "./Retail_Transactions_Dataset.csv"
TIME_COL = "Date"
VALUE_COL = "Total_Cost"
PERIODS = 6  # forecast horizon (months)
OUTFILE = "forecast_total_cost.png"

def main():
    # 1) Load and clean dataset
    df = pd.read_csv(CSV_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    print(f"✅ Loaded dataset with {len(df):,} rows")

    # 2) Run forecast
    agent = ForecastAgent()
    result = agent.generate_forecast(
        data=df,
        periods=PERIODS,
        time_col=TIME_COL,
        value_cols=[VALUE_COL]
    )

    # 3) Print summary (the part you requested)
    series = result.get(VALUE_COL, {})
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

    # 4) Exit if forecast failed
    if series.get("status") != "ok":
        print("❌ Forecast failed or insufficient data, aborting plot.")
        return

    # 5) Prepare historical actuals
    actual = (
        df[[TIME_COL, VALUE_COL]]
        .rename(columns={TIME_COL: "ds", VALUE_COL: "y"})
        .set_index("ds").sort_index()
        .resample("MS").sum()
        .reset_index()
    )

    # 6) Prepare forecast data
    fc = pd.DataFrame(series["forecast"])
    fc["ds"] = pd.to_datetime(fc["ds"])
    last_actual = series["last_actual"]
    last_pt = pd.DataFrame([{
        "ds": pd.to_datetime(last_actual["ds"]),
        "yhat": last_actual["y"],
        "yhat_lower": last_actual["y"],
        "yhat_upper": last_actual["y"],
    }])
    fc_plot = pd.concat([last_pt, fc], ignore_index=True)

    # 7) Plot
    plt.figure(figsize=(10, 5))
    plt.plot(actual["ds"], actual["y"], label="Actual (Monthly)", linewidth=2)
    plt.plot(fc_plot["ds"], fc_plot["yhat"], label="Forecast (yhat)", linewidth=2, color="tab:orange")
    plt.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"], color="orange", alpha=0.2, label="95% interval")
    plt.title("Total Cost Forecast (Monthly)")
    plt.xlabel("Date")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=200)
    print(f"\n📊 Forecast chart saved to {OUTFILE}")


    # 8) Show last 3 points for convenience
    print("\nLast 6 forecasted points:")
    print(fc.tail(6)[["ds", "yhat", "yhat_lower", "yhat_upper"]])

if __name__ == "__main__":
    main()
