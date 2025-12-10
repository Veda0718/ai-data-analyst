# AI Data Analyst - Automated Data-to-Insight Agent

A lightweight AI-powered analytics system that ingests structured sales data (CSV or Excel), automatically computes key business insights, detects trends and anomalies, and produces actionable recommendations.
Users can ask natural-language questions (“Which country generated the most revenue?”), generate dynamic visualizations, and receive an executive summary through an interactive Streamlit dashboard or a FastAPI backend.

## Key Features:
 - Automated Analysis - Upload data, get insights instantly
 - AI-Powered Visualizations - Adaptive charts that understand your data
 - Chat with Your Data - Ask questions in natural language
 - On-Demand Plotting - Request custom charts through conversation
 - Executive Summaries - AI-generated business narratives
 - Anomaly Detection - Automatically identify unusual patterns
 - Export Reports - Download insights in JSON or CSV

## Tech Stack
Python, OpenAI, FastAPI, Streamlit, Langchain

## Project Structure
```bash
.
├── app.py                # Streamlit frontend (UI, visualizations, chat interface)
├── api.py                # FastAPI backend exposing /analyze and /query endpoints
├── analyzer.py           # Core insight engine: trends, summaries, anomalies, recommendations
├── query_helper.py       # Standalone NL query system (fallback when no LLM available)
├── requirements.txt
└── README.md
```

## Implementation
### 1. Clone the Repository
```bash
git clone https://github.com/Veda0718/ai-data-analyst.git
cd ai-data-analyst
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure OpenAI API Key
Create a .env file:
```bash
OPENAI_API_KEY=your_key_here
```

 ### 5. Run Streamlit UI
```bash
streamlit run app.py
```

### 6. Run FastAPI Backend
```bash
uvicorn api:app --reload
```

