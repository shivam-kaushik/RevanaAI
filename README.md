# Revana - Agentic Sales Assistant

**Revana** is an advanced, AI-powered analytics assistant designed to democratize data access for retail and sales teams. Instead of relying on static dashboards or SQL experts, users can chat with their data in plain English.

The system uses a **Multi-Agent Architecture** to plan, route, and execute complex analytical tasks—from simple SQL queries to advanced forecasting and anomaly detection.

![Architecture Diagram](https://mermaid.ink/img/pako:eNp1ksFu2zAMhl8F_bIFEmzJ6WEF0m1Dhq0osA3rDjuIoi22FkWqJB9zCPzuko-TFAWy-STyR_4S-Qo1K4USWn-o6h_GgR58e2iM6_f7r3B8gu_vH-Dw8AjH4wlOJyhOIAUoQAmKUDw8wAnK4yMchXyC4nCCUygfHuF4hPLh_v4BjuMHOB7v4RSKx_sHOP0ExSmUj_dweIBSKB7gBIVQPDzA8XCCUygfT3A6QClfT1A-3N8_wAkK-XCCUygfH-EUyscTnEIpX4Xy8QSlUD48wPEBSvn6AcqH-_sHOEExnMKJFE_RFNosC2XWpTRG18oY_WOsMdo0X-rG1E2V46LzNq_zMvO5dI029Vq3WpumkMY0ZVZknbe5rOq8zK3W-r3yG_n_vM6r3GuT13ld1Lks67zOq6LOy9xrnad8neep2OQpX-e5qvMy99rkdV4XdR7yNc9VXZdF521e51K3ZVm3eZUX_TzP40JqU--P-R9T-p_y?type=png) 
*(Note: Diagram is conceptual logic)*

## 🚀 Key Features

*   **Natural Language to SQL**: Ask "What were the total sales last month?" and get an accurate answer backed by real data.
*   **Time-Series Forecasting**: Ask "Predict revenue for the next 6 months," and the system uses **Facebook Prophet** to generate reliable forecasts.
*   **Anomaly Detection**: Automatically detects outliers, spikes, and drops in sales data using statistical methods (Z-Score, IQR).
*   **Semantic Search**: Find "best performing products" or "similar customers" using vector embeddings (`pgvector`) search.
*   **Dynamic Visualizations**: Auto-generates Line charts, Bar graphs, and Pie charts based on your query context.
*   **Hybrid AI**: Uses a local **Qwen** model for fast Text-to-SQL generation, falling back to **GPT-4o/3.5** for complex reasoning.

## 🛠️ Technology Stack

*   **Backend**: Python, FastAPI
*   **Frontend**: HTML5, Vanilla JS, Jinja2 Templates
*   **Database**: PostgreSQL + pgvector
*   **AI Framework**: LangChain
*   **LLMs**: OpenAI (GPT-4o-mini, GPT-3.5-turbo), Local Qwen (via `llama-cpp-python`)
*   **Analytics**: Pandas, NumPy, Scikit-learn, Facebook Prophet
*   **Visualization**: Matplotlib, Plotly.js, Chart.js

## 📦 Installation

### Prerequisites
*   Python 3.10+
*   PostgreSQL installed and running
*   OpenAI API Key

### Setup Steps

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/revana.git
    cd revana
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment**:
    Create a `.env` file in the root directory (or use `backend/config.py`) and set your keys:
    ```bash
    DATABASE_URL=postgresql://user:password@localhost:5432/revana_db
    OPENAI_API_KEY=sk-...
    ```

4.  **Initialize the Database**:
    Ensure your Postgres instance is running. The application handles table creation dynamically when you upload a CSV.

## 🏃 Usage

1.  **Start the Server**:
    Run the startup script from the root directory:
    ```bash
    python run.py
    ```
    The server will start at `http://localhost:8000`.

2.  **Access the UI**:
    Open your browser and navigate to `http://localhost:8000`.

3.  **Upload Data**:
    *   Click the **Upload CSV** button.
    *   Select your sales dataset (must contain headers like `Date`, `Sales`, `Product`, etc.).
    *   The system will automatically create a table and index it for vector search.

4.  **Start Chatting**:
    *   *Analytic*: "Show me sales by week for 2023."
    *   *Forecast*: "Forecast sales for the next 3 months."
    *   *Anomaly*: "Are there any unusual spikes in revenue?"
    *   *Search*: "Find products related to summer."

## 🧩 Architecture Overview

The system follows a **Planner-Executor** pattern:

1.  **User Request**: Received by `/chat` endpoint.
2.  **Planner Agent**: Analyzes intent (Data Query vs. Greeting vs. Forecast).
3.  **Routing**:
    *   **SQL Agent**: Generates SQL for database queries.
    *   **Vector Agent**: Handles semantic similarity search.
    *   **Forecast Agent**: specialized pipeline for time-series.
    *   **Anomaly Agent**: Statistical analysis on SQL results.
4.  **Aggregation**: The **Analysis Agent** (Insight Agent) generates a textual summary and charts.
5.  **Response**: Frontend renders the text, markdown tables, and plots.

## 🤝 Contributing

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
