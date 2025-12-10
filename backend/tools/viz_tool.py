# backend/tools/viz_tool.py
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

class VizTool:
    def __init__(self, output_dir="frontend/static/forecast"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
  
    # viz_tool.py
    def _web_path(self, p: str) -> str:
        p = p.replace("\\", "/")
        if "frontend/static/" in p:
            return p.split("frontend/", 1)[1]   # -> 'static/forecast/....png'
        if "static/" in p:
            return p[p.index("static/"):]       # fallback
        return p

    def plot_future_only(
        self,
        history: pd.DataFrame,
        forecast: pd.DataFrame,
        tag: str,
        title: str,
        y_label: str = "Total Sales",
        dpi: int = 150
    ):
        """
        Draw only the forecast horizon with uncertainty band.
        history: monthly ['ds','y'] (used only to find last actual date)
        forecast: Prophet output containing both history+future; we will slice future only.
        """
        # last actual point
        last_ds = pd.to_datetime(history['ds']).max()

        # FUTURE slice
        fc = forecast[forecast['ds'] > last_ds].copy()
        if fc.empty:
            # As a safeguard, fall back to last N rows if slicing failed
            fc = forecast.tail(12).copy()

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(fc['ds'], fc['yhat'], label='Forecast (yhat)', linewidth=2)
        ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'],
                        alpha=0.25, label='95% interval')

        # cosmetics
        ax.set_title(title, pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel(y_label)
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.grid(alpha=0.25)
        ax.legend(loc='upper left')

        # save
        save_path = os.path.join(self.output_dir, f"future_{tag}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi)
        plt.close(fig)
        return self._web_path(save_path)

    def plot_combined(self, hist_df, fcst_df, tag,
                      title="Total Sales Forecast (Monthly)",
                      ylabel="Total Sales"):
        """
        One figure with full history (blue) and forecast continuation (orange)
        starting exactly at the last historical month, with a 95% band.
        hist_df: columns ['ds','y'] monthly
        fcst_df: columns ['ds','yhat','yhat_lower','yhat_upper'] monthly
        """
        # sort & ensure datetime
        hist_df = hist_df.copy()
        hist_df["ds"] = pd.to_datetime(hist_df["ds"])
        hist_df = hist_df.sort_values("ds")

        fcst_df = fcst_df.copy()
        fcst_df["ds"] = pd.to_datetime(fcst_df["ds"])
        fcst_df = fcst_df.sort_values("ds")

        last_ds = hist_df["ds"].max()
        last_y  = float(hist_df.loc[hist_df["ds"] == last_ds, "y"].iloc[0])

        # Build the forecast line so it *continues* from the last actual point
        fc_future = fcst_df[fcst_df["ds"] > last_ds]
        fc_line = pd.concat([
            pd.DataFrame([{
                "ds": last_ds,
                "yhat": last_y,
                "yhat_lower": last_y,
                "yhat_upper": last_y
            }]),
            fc_future[["ds","yhat","yhat_lower","yhat_upper"]]
        ], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # 1) Full history
        ax.plot(hist_df["ds"], hist_df["y"], linewidth=2, label="Actual (Monthly)")

        # 2) Forecast (orange) from the last actual forward
        ax.plot(fc_line["ds"], fc_line["yhat"], linewidth=3, label="Forecast (yhat)", color="tab:orange")

        # 3) 95% confidence band only for future months
        if not fc_future.empty:
            ax.fill_between(fc_future["ds"], fc_future["yhat_lower"], fc_future["yhat_upper"],
                            alpha=0.20, label="95% interval", color="tab:orange")

        # A small marker where the forecast starts (optional)
        ax.axvline(last_ds, linestyle="--", linewidth=1, alpha=0.5)

        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.set_xlim(hist_df["ds"].min(), (fcst_df["ds"].max()))
        ax.legend()
        fig.tight_layout()

        out = os.path.join(self.output_dir, f"combined_{tag}.png")
        fig.savefig(out, dpi=100)
        plt.close(fig)
        return self._web_path(out)
    
    
     
