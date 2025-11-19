# backend/agents/forecast_agent.py
from __future__ import annotations
import os, re
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any
from sqlalchemy import create_engine, text
import base64 
from backend.tools import SQLTool, NL2SQLTool, ProphetTool, VizTool, SummaryTool



@dataclass
class ForecastAgentConfig:
    database_url: str
    schema_text: str
    output_dir: str = "frontend/static/forecast"
    default_horizon: int = 3


class ForecastAgent:
    """Independent Forecast agent:
    NL -> SQL -> fetch -> (ds,y) -> Prophet -> summary -> plots
    """
    def __init__(self, cfg: ForecastAgentConfig, llm):
        self.cfg = cfg
        self.engine = create_engine(cfg.database_url, future=True)
        self.sql = SQLTool(self.engine)
        self.nl2sql = NL2SQLTool(llm=llm, schema_text=cfg.schema_text)
        self.prophet = ProphetTool()
        self.viz = VizTool(output_dir=cfg.output_dir)
        self.summary = SummaryTool(llm=llm)
        
    def run(self, nl_query: str) -> Dict[str, Any]:
        horizon = self._parse_horizon(nl_query) or self.cfg.default_horizon

        # ⬇️ figure out active table
        if "ACTIVE_TABLE:" not in self.cfg.schema_text:
            return {
                "status": "error",
                "message": "Schema context missing ACTIVE_TABLE. Rebuild schema_text and re-init ForecastAgent."
            }
        active_line = next((ln for ln in self.cfg.schema_text.splitlines() if ln.startswith("ACTIVE_TABLE:")), "")
        print(f"[ForecastAgent] Using {active_line}")  # shows ACTIVE_TABLE in logs

        # 1) NL -> SQL
        sql = self.nl2sql.run(nl_query)

        # 2) Execute SQL
        df = self.sql.run(sql)
        if df.empty:
            return {"status": "empty", "query": nl_query, "sql": sql, "message": "No rows returned."}

        # 3) Ensure (ds, y)
        df = self._coerce_to_ts(df)

        # 4) Forecast
        model, fcst = self.prophet.run(df, horizon=horizon)

        # 5) Plots
        #tag = self._slug(nl_query)[:40]
        #fcst_path = self.viz.plot_forecast(fcst, tag)
        # backend/agents/forecast_agent.py (inside run, after fcst is computed)
        tag = self._slug(nl_query)[:40]
        combined_png = self.viz.plot_combined(df, fcst, tag)
        future_path = self.viz.plot_future_only(
            df,
            fcst[["ds","yhat","yhat_lower","yhat_upper"]].copy(),
            tag=tag,
            title = "Forecast (Monthly)",
            y_label= "Total Sales",
            dpi = 150
        )
        combined_path = self.viz.plot_combined(
            hist_df=df,
            fcst_df=fcst[["ds","yhat","yhat_lower","yhat_upper"]].copy(),
            tag=tag,
            title="Total Sales Forecast (Monthly)",
            ylabel="Total Sales"
        )
        # Convert the saved PNGs to base64 so frontend can use them directly
        #hist_b64 = self._png_to_base64(hist_path)
        combined_b64 = self._png_to_base64(combined_path)
        future_b64 = self._png_to_base64(future_path)

        print("DEBUG combined_path =", combined_path)
        # 6) Summary
        text_summary = self.summary.run(nl_query, fcst, horizon)

        markdown = (
            f"🔮 Forecast ({horizon} months)\n"
            f"![Forecast]({combined_b64})\n\n"
            f"![Forecast]({future_b64})\n\n"
            f"**Summary:** {text_summary}\n"
        )

        return {
            "status": "ok",
            "query": nl_query,
            "sql": sql,
            "horizon": horizon,
            "plots": {
                #"combined_png": combined_path,
                "combined_base64": combined_b64,
                "future_base64": future_b64,  
            },
            "markdown": markdown,
            "df_head": df.head(10).to_dict("records"),
            "forecast_tail": fcst.tail(horizon).to_dict("records"),
        }

 
    def refresh_schema(self, schema_text: str):
        self.cfg.schema_text = schema_text
        self.nl2sql.update_schema(schema_text)   

    # ---------- helpers ----------
    @staticmethod
    def _parse_horizon(q: str) -> int | None:
        m = re.search(r'(?:next|for)\s+(\d{1,2})\s+month', q, re.I)
        return int(m.group(1)) if m else None

    @staticmethod
    def _slug(text: str) -> str:
        return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

    @staticmethod
    def _coerce_to_ts(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c.lower(): c for c in df.columns}
        if "ds" in cols and "y" in cols:
            ds, y = cols["ds"], cols["y"]
            out = df[[ds, y]].rename(columns={ds: "ds", y: "y"})
        else:
            date_col = next((c for c in df.columns if "date" in c.lower() or "month" in c.lower()), None)
            value_col = next((c for c in df.columns if c != date_col), None)
            out = df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})

        out["ds"] = pd.to_datetime(out["ds"])
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
        out = out.dropna(subset=["ds", "y"]).sort_values("ds")
        out["ds"] = out["ds"].values.astype("datetime64[M]")  # monthly
        return out[["ds", "y"]]
    
    def _png_to_base64(self, web_path: str) -> str:
        """
        Convert a PNG file (referenced by the web path like 'static/forecast/xxx.png')
        into a data:image/png;base64,... string.
        """
        # web_path looks like: 'static/forecast/combined_predict-next-6-months.png'
        filename = os.path.basename(web_path)
        fs_path = os.path.join(self.cfg.output_dir, filename)  # e.g. frontend/static/forecast/...

        with open(fs_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return "data:image/png;base64," + encoded
