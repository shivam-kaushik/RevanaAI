class SummaryTool:
    """Summarizes the forecast result using an LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run(self, nl_query: str, forecast_df, horizon: int) -> str:
        sample = forecast_df.tail(horizon)[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_string(index=False)
        prompt = f"""
Summarize this forecast for a business audience in a short, clear paragraph.
Query: {nl_query}
Forecast (last {horizon} rows):
{sample}
"""
        response = self.llm.invoke(prompt)
        return response.content.strip()
