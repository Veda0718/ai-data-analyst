from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
from analyzer import DataAnalyzer
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from datetime import datetime, date

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def json_serializer(obj):
    """JSON serializer for objects not serializable by default"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")

app = FastAPI(
    title="AI Data Analyst API",
    description="Backend API for automated data analysis and insights generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Data Analyst API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Upload file for analysis",
            "POST /query": "Natural language query",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Data Analyst"}

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    enable_anomaly: bool = True,
    enable_ai_narrative: bool = True
):
    """
    Analyze uploaded CSV or Excel file
    
    Parameters:
    - file: CSV or Excel file
    - enable_anomaly: Enable anomaly detection
    - enable_ai_narrative: Generate AI narrative
    
    Returns:
    - insights: List of insights
    - recommendations: List of recommendations
    - summary: Dataset summary
    - narrative: AI-generated narrative (if enabled)
    """
    try:
        # Read file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if df is None:
                raise HTTPException(status_code=400, detail="Unable to decode CSV file. Please check encoding.")
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Initialize analyzer
        analyzer = DataAnalyzer(df)
        df = analyzer.preprocess_data()
        
        # Generate insights
        insights = analyzer.generate_insights()
        recommendations = analyzer.generate_recommendations(insights)
        summary = analyzer.generate_summary_stats()
        
        # Generate AI narrative
        narrative = None
        if enable_ai_narrative and openai_client:
            try:
                prompt = f"""
                As a senior business analyst, provide a concise executive summary:
                
                Dataset Overview: {json.dumps(summary, default=json_serializer, indent=2)}
                Key Insights: {json.dumps(insights, default=json_serializer, indent=2)}
                Recommendations: {json.dumps(recommendations, default=json_serializer, indent=2)}
                
                Write a 3-4 paragraph executive summary highlighting critical findings,
                business implications, and recommended actions.
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an expert business analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                narrative = response.choices[0].message.content
            except Exception as e:
                narrative = f"AI narrative unavailable: {str(e)}"
        
        return {
            "status": "success",
            "file_name": file.filename,
            "insights": insights,
            "recommendations": recommendations,
            "summary": summary,
            "narrative": narrative,
            "metadata": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": df.columns.tolist()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/query")
async def natural_language_query(
    query: str,
    file: UploadFile = File(...)
):
    """
    Answer natural language questions about the dataset
    
    Parameters:
    - query: Natural language question
    - file: CSV or Excel file
    
    Returns:
    - answer: AI-generated answer
    """
    try:
        # Read file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Generate answer
        if not openai_client:
            raise HTTPException(status_code=503, detail="OpenAI API key not configured")
        
        df_info = df.describe().to_string()
        
        prompt = f"""
        Dataset Summary:
        {df_info}
        
        Columns: {', '.join(df.columns)}
        
        Question: {query}
        
        Provide a specific, data-driven answer.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        
        return {
            "status": "success",
            "query": query,
            "answer": response.choices[0].message.content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.get("/sample-data")
async def get_sample_data():
    """
    Return information about sample datasets
    """
    return {
        "available_samples": [
            {
                "name": "sales_data.csv",
                "description": "Weekly sales data by region and product category",
                "columns": ["Date", "Region", "Product_Category", "Revenue", "Units_Sold", "Customer_Count"]
            }
        ],
        "note": "Upload your own dataset using the /analyze endpoint"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)