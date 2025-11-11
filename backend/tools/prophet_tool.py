from prophet import Prophet

class ProphetTool:
    """Fits Prophet models and generates forecasts."""

    def __init__(self):
        self.model = None

    def run(self, df, horizon: int = 3):
        if "ds" not in df or "y" not in df:
            raise ValueError("DataFrame must have columns 'ds' and 'y'")
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=horizon, freq="MS")
        forecast = model.predict(future)
        return model, forecast
