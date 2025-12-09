import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import DataAnalyzer
from query_helper import DataQueryHelper
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import numpy as np
from datetime import datetime, date
import tempfile

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# LangChain imports (try to import, graceful fallback)
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
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

def safe_json_dumps(data):
    """Safely convert data to JSON string"""
    return json.dumps(data, indent=2, default=json_serializer)

def generate_ai_narrative(insights, recommendations, df_summary):
    """Generate natural language narrative using OpenAI"""
    if not openai_client:
        return "AI narrative unavailable. Please set OPENAI_API_KEY in your .env file to enable this feature."
    
    try:
        # Convert data to JSON-safe format
        prompt = f"""
        As a senior business analyst, provide a concise executive summary based on this data analysis:
        
        Dataset Overview:
        {safe_json_dumps(df_summary)}
        
        Key Insights:
        {safe_json_dumps(insights)}
        
        Recommendations:
        {safe_json_dumps(recommendations)}
        
        Write a 3-4 paragraph executive summary that:
        1. Highlights the most critical findings
        2. Explains business implications
        3. Emphasizes recommended actions
        
        Use clear, professional business language.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst providing executive insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI narrative generation unavailable. Error: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .insight-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #2ca02c;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

def create_plot_from_question(df, question):
    """Create a plot based on natural language question"""
    question_lower = question.lower()
    
    try:
        # Extract column names mentioned in the question
        mentioned_cols = []
        for col in df.columns:
            if col.lower() in question_lower:
                mentioned_cols.append(col)
        
        # Identify plot type
        if any(word in question_lower for word in ['trend', 'over time', 'time series', 'line']):
            # Time series plot
            date_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                date_col = date_cols[0]
                
                # Use mentioned numeric column or first numeric
                metric_col = next((col for col in mentioned_cols if col in numeric_cols), numeric_cols[0])
                
                # Check for grouping
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                mentioned_cat = next((col for col in mentioned_cols if col in categorical_cols), None)
                
                if mentioned_cat or len(categorical_cols) > 0:
                    group_col = mentioned_cat or categorical_cols[0]
                    fig = px.line(
                        df.sort_values(date_col),
                        x=date_col,
                        y=metric_col,
                        color=group_col,
                        title=f'{metric_col} Trend by {group_col}',
                        markers=True
                    )
                else:
                    fig = px.line(
                        df.sort_values(date_col),
                        x=date_col,
                        y=metric_col,
                        title=f'{metric_col} Over Time',
                        markers=True
                    )
                
                return fig, f"Created line chart showing {metric_col} over time"
        
        elif any(word in question_lower for word in ['bar', 'compare', 'comparison', 'by']):
            # Bar chart
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Smart column selection based on question
                cat_col = None
                metric_col = None
                
                # Look for mentioned columns
                for col in mentioned_cols:
                    if col in categorical_cols and not cat_col:
                        cat_col = col
                    if col in numeric_cols and not metric_col:
                        metric_col = col
                
                # Fallback: prefer meaningful columns
                if not cat_col:
                    # Avoid ID-like columns for categories
                    good_cats = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', 'number', 'code'])]
                    cat_col = good_cats[0] if good_cats else categorical_cols[0]
                
                if not metric_col:
                    # Prefer amount/value/sales columns
                    priority_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'total', 'quantity'])]
                    metric_col = priority_cols[0] if priority_cols else numeric_cols[0]
                
                grouped = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=grouped.index,
                    y=grouped.values,
                    title=f'{metric_col} by {cat_col}',
                    labels={'x': cat_col, 'y': metric_col}
                )
                
                return fig, f"Created bar chart comparing {metric_col} across {cat_col}"
        
        elif any(word in question_lower for word in ['distribution', 'histogram', 'spread']):
            # Histogram
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) > 0:
                # Use mentioned column or prefer meaningful columns
                metric_col = next((col for col in mentioned_cols if col in numeric_cols), None)
                if not metric_col:
                    priority_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'quantity'])]
                    metric_col = priority_cols[0] if priority_cols else numeric_cols[0]
                
                fig = px.histogram(
                    df,
                    x=metric_col,
                    title=f'Distribution of {metric_col}',
                    nbins=30
                )
                
                return fig, f"Created histogram showing distribution of {metric_col}"
        
        elif any(word in question_lower for word in ['scatter', 'correlation', 'relationship']):
            # Scatter plot
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Use mentioned columns or prefer meaningful ones
                mentioned_numeric = [col for col in mentioned_cols if col in numeric_cols]
                if len(mentioned_numeric) >= 2:
                    x_col, y_col = mentioned_numeric[0], mentioned_numeric[1]
                else:
                    x_col = numeric_cols[0]
                    y_col = numeric_cols[1]
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f'{y_col} vs {x_col}',
                    trendline='ols'
                )
                
                return fig, f"Created scatter plot showing relationship between {x_col} and {y_col}"
        
        # Default: bar chart of meaningful columns
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Use mentioned columns or smart selection
            cat_col = next((col for col in mentioned_cols if col in categorical_cols), None)
            metric_col = next((col for col in mentioned_cols if col in numeric_cols), None)
            
            if not cat_col:
                good_cats = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', 'number', 'code'])]
                cat_col = good_cats[0] if good_cats else categorical_cols[0]
            
            if not metric_col:
                priority_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'total', 'quantity'])]
                metric_col = priority_cols[0] if priority_cols else numeric_cols[0]
            
            grouped = df.groupby(cat_col)[metric_col].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=grouped.index,
                y=grouped.values,
                title=f'Top 10 {cat_col} by {metric_col}',
                labels={'x': cat_col, 'y': metric_col}
            )
            
            return fig, f"Created chart based on your data"
        
        return None, "Could not determine appropriate chart type from your question."
        
    except Exception as e:
        return None, f"Error creating plot: {str(e)}"

def answer_data_question(df, question):
    """Answer questions about data using multiple methods"""
    
    # Check if this is a plotting request - handle it specially
    is_plot_request = any(word in question.lower() for word in ['plot', 'chart', 'graph', 'visualize', 'show me', 'draw', 'create'])
    
    if is_plot_request:
        # ALWAYS use our smart plotting function for plots
        fig, message = create_plot_from_question(df, question)
        if fig:
            return (fig, message), None
        else:
            return None, message
    
    # For non-plot questions, use LangChain or fallback
    # Method 1: Try LangChain CSV Agent (most powerful)
    if LANGCHAIN_AVAILABLE and openai_client and os.getenv('OPENAI_API_KEY'):
        try:
            # Create LangChain agent
            llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Enhanced prefix - explicitly tell it NOT to create plots
            prefix = """
You are a data analyst assistant working with a pandas dataframe. 

When answering questions:
1. Provide SPECIFIC numbers and values from the data
2. Give DIRECT answers (e.g., "Madrid ordered 1,020 vintage cars")
3. Show the top 3-5 results when ranking
4. Format numbers with commas for readability
5. Be concise but complete

IMPORTANT: Do NOT create plots or visualizations. Only analyze data and provide numerical answers.

Available tools: python_repl_ast for executing Python code.

"""
            
            agent = create_pandas_dataframe_agent(
                llm,
                df,
                verbose=False,
                agent_type="openai-tools",
                allow_dangerous_code=True,
                prefix=prefix,
                handle_parsing_errors=True
            )
            
            # Enhanced question for better answers
            enhanced_question = f"""
{question}

Remember:
- Provide specific numbers from the actual data
- Be direct and clear
- Format output nicely
- Do NOT create any plots or visualizations
"""
            
            # Get answer from agent
            response = agent.invoke(enhanced_question)
            
            # Extract the answer
            if isinstance(response, dict):
                answer = response.get('output', str(response))
            else:
                answer = str(response)
            
            return answer, None
            
        except Exception as e:
            # Fall back to method 2 if LangChain fails
            pass
    
    # Method 2: Use custom query helper (no API key needed)
    try:
        query_helper = DataQueryHelper(df)
        data_result = query_helper.query(question)
        data_answer = query_helper.format_result(data_result)
        
        # If we have OpenAI but LangChain not available, enhance with OpenAI
        if openai_client and not LANGCHAIN_AVAILABLE:
            try:
                # Get enhanced natural language answer
                prompt = f"""Based on this data analysis, provide a clear, natural answer:

Question: {question}

Data Analysis:
{data_answer}

Provide a conversational response (2-3 sentences) that directly answers the question with specific numbers."""
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.3
                )
                
                ai_answer = response.choices[0].message.content
                return f"{ai_answer}\n\n{data_answer}", None
            except:
                pass
        
        return data_answer, None
        
    except Exception as e:
        return None, f"Error analyzing data: {str(e)}"
    """Generate natural language narrative using OpenAI"""
    if not openai_client:
        return "AI narrative unavailable. Please set OPENAI_API_KEY in your .env file."
    
    try:
        # Convert data to JSON-safe format
        prompt = f"""
        As a senior business analyst, provide a concise executive summary based on this data analysis:
        
        Dataset Overview:
        {safe_json_dumps(df_summary)}
        
        Key Insights:
        {safe_json_dumps(insights)}
        
        Recommendations:
        {safe_json_dumps(recommendations)}
        
        Write a 3-4 paragraph executive summary that:
        1. Highlights the most critical findings
        2. Explains business implications
        3. Emphasizes recommended actions
        
        Use clear, professional business language.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst providing executive insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"AI narrative generation unavailable. Error: {str(e)}"

def generate_dynamic_visualizations(df, insights, openai_client):
    """
    Dynamically generate visualizations using AI to understand data structure.
    No hardcoding - adapts to any dataset.
    """
    
    if not openai_client:
        return generate_smart_fallback_visualizations(df)
    
    try:
        # Prepare comprehensive data analysis for AI
        data_profile = {
            'columns': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
            'sample_values': {
                col: df[col].dropna().head(3).tolist() 
                for col in df.columns[:10]  # First 10 columns
            },
            'column_stats': {
                col: {
                    'unique_count': int(df[col].nunique()),
                    'null_count': int(df[col].isnull().sum()),
                    'sample_values': df[col].dropna().unique()[:5].tolist()
                }
                for col in df.columns[:15]  # Stats for first 15 columns
            },
            'insights_summary': [
                {'title': i['title'], 'content': i['content']} 
                for i in insights[:3]
            ]
        }
        
        prompt = f"""
You are a data visualization expert. Analyze this dataset and recommend 4-6 visualizations that would provide the most business value.

Dataset Profile:
{json.dumps(data_profile, indent=2, default=str)}

Instructions:
1. Analyze the data structure and understand what kind of business/domain this data represents
2. Recommend visualizations that would answer key business questions
3. Use actual column names from the dataset
4. Choose appropriate chart types based on data types
5. Provide clear titles and business purposes

Chart Type Guidelines:
- Use "line" for time-series trends (requires datetime column)
- Use "bar" for category comparisons (categorical vs numeric)
- Use "horizontal_bar" for ranking/top N (when labels are long)
- Use "histogram" for distribution analysis (single numeric column)
- Use "scatter" for correlation analysis (two numeric columns)
- Use "box" for statistical distribution (numeric column, optionally grouped by categorical)

Respond with a JSON array of 4-6 visualization specifications:

[
  {{
    "chart_type": "line",
    "x_column": "exact_column_name",
    "y_column": "exact_column_name",
    "color_by": "optional_grouping_column",
    "title": "Clear descriptive title",
    "business_question": "What business question does this answer?",
    "aggregation": "sum|mean|count|none"
  }}
]

Rules:
- Only use columns that exist in the dataset
- Ensure chart type matches data types (e.g., line charts need datetime)
- Aggregation: "sum" for totals, "mean" for averages, "count" for frequencies, "none" for raw data
- Make titles business-friendly, not technical
- Each chart should answer a different business question

Respond ONLY with valid JSON array. No explanation, just JSON.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business intelligence expert specializing in data visualization. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        
        # Parse AI response
        response_text = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        if '```' in response_text:
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        viz_specs = json.loads(response_text)
        
        # Generate charts from AI specifications
        charts = []
        for spec in viz_specs[:6]:  # Max 6 charts
            try:
                fig = create_dynamic_chart(df, spec)
                if fig:
                    charts.append({
                        'figure': fig,
                        'title': spec.get('title', 'Chart'),
                        'purpose': spec.get('business_question', 'Analysis visualization')
                    })
            except Exception as e:
                st.warning(f"Could not create chart '{spec.get('title', 'Unknown')}': {str(e)}")
        
        if charts:
            return charts
        else:
            st.warning("AI recommendations could not be visualized. Using smart fallback.")
            return generate_smart_fallback_visualizations(df)
            
    except Exception as e:
        st.warning(f"AI visualization generation failed: {str(e)}. Using fallback visualizations.")
        return generate_smart_fallback_visualizations(df)

def create_dynamic_chart(df, spec):
    """
    Create a chart dynamically based on AI specification.
    Handles aggregation and various chart types.
    """
    chart_type = spec.get('chart_type', '').lower()
    x_col = spec.get('x_column')
    y_col = spec.get('y_column')
    color_by = spec.get('color_by')
    title = spec.get('title', 'Chart')
    aggregation = spec.get('aggregation', 'none')
    
    # Validate columns exist
    if x_col and x_col not in df.columns:
        return None
    if y_col and y_col not in df.columns:
        return None
    if color_by and color_by not in df.columns:
        color_by = None
    
    try:
        # Prepare data based on aggregation
        if aggregation != 'none' and x_col and y_col:
            if color_by:
                if aggregation == 'sum':
                    plot_data = df.groupby([x_col, color_by])[y_col].sum().reset_index()
                elif aggregation == 'mean':
                    plot_data = df.groupby([x_col, color_by])[y_col].mean().reset_index()
                elif aggregation == 'count':
                    plot_data = df.groupby([x_col, color_by])[y_col].count().reset_index()
                else:
                    plot_data = df
            else:
                if aggregation == 'sum':
                    plot_data = df.groupby(x_col)[y_col].sum().reset_index()
                elif aggregation == 'mean':
                    plot_data = df.groupby(x_col)[y_col].mean().reset_index()
                elif aggregation == 'count':
                    plot_data = df.groupby(x_col)[y_col].count().reset_index()
                else:
                    plot_data = df
        else:
            plot_data = df
        
        # Limit data for readability
        if chart_type in ['bar', 'horizontal_bar'] and len(plot_data) > 20:
            if y_col:
                plot_data = plot_data.nlargest(20, y_col)
            else:
                plot_data = plot_data.head(20)
        
        # Create chart based on type
        if chart_type == 'line':
            fig = px.line(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_by,
                title=title,
                markers=True
            )
            fig.update_layout(height=400, hovermode='x unified')
            
        elif chart_type == 'bar':
            if color_by:
                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    color=color_by,
                    title=title
                )
            else:
                fig = px.bar(
                    plot_data,
                    x=x_col,
                    y=y_col,
                    title=title
                )
            fig.update_layout(height=400)
            
        elif chart_type == 'horizontal_bar':
            fig = px.bar(
                plot_data,
                x=y_col,
                y=x_col,
                orientation='h',
                title=title,
                color=color_by
            )
            fig.update_layout(height=max(400, len(plot_data) * 25))
            
        elif chart_type == 'histogram':
            fig = px.histogram(
                df,
                x=x_col,
                title=title,
                nbins=30,
                color=color_by
            )
            fig.update_layout(height=400)
            
        elif chart_type == 'scatter':
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_by,
                title=title,
                trendline='ols' if not color_by else None
            )
            fig.update_layout(height=400)
            
        elif chart_type == 'box':
            fig = px.box(
                df,
                x=color_by if color_by else None,
                y=x_col,
                title=title
            )
            fig.update_layout(height=400)
            
        else:
            return None
        
        return fig
        
    except Exception as e:
        return None

def generate_smart_fallback_visualizations(df):
    """
    Smart fallback when AI is unavailable.
    Uses data structure analysis to create relevant charts.
    """
    charts = []
    
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Find most relevant columns (avoid IDs)
    good_numeric = [c for c in numeric_cols if not any(x in c.lower() for x in ['id', 'number', 'code'])]
    good_categorical = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', 'number', 'code'])]
    
    metric_col = good_numeric[0] if good_numeric else (numeric_cols[0] if numeric_cols else None)
    category_col = good_categorical[0] if good_categorical else (categorical_cols[0] if categorical_cols else None)
    date_col = date_cols[0] if date_cols else None
    
    # 1. Time series if date available
    if date_col and metric_col:
        try:
            daily_data = df.groupby(date_col)[metric_col].sum().reset_index()
            fig = px.line(daily_data, x=date_col, y=metric_col, 
                         title=f'{metric_col} Over Time', markers=True)
            fig.update_layout(height=400)
            charts.append({
                'figure': fig,
                'title': f'{metric_col} Trend',
                'purpose': 'Track performance over time'
            })
        except:
            pass
    
    # 2. Category comparison if categorical available
    if category_col and metric_col:
        try:
            cat_data = df.groupby(category_col)[metric_col].sum().sort_values(ascending=False).head(15)
            fig = px.bar(x=cat_data.index, y=cat_data.values,
                        title=f'{metric_col} by {category_col}',
                        labels={'x': category_col, 'y': metric_col})
            fig.update_layout(height=400)
            charts.append({
                'figure': fig,
                'title': f'{metric_col} by {category_col}',
                'purpose': 'Compare across categories'
            })
        except:
            pass
    
    # 3. Distribution if numeric available
    if metric_col:
        try:
            fig = px.histogram(df, x=metric_col, 
                             title=f'Distribution of {metric_col}', nbins=30)
            fig.update_layout(height=400)
            charts.append({
                'figure': fig,
                'title': f'{metric_col} Distribution',
                'purpose': 'Understand value spread'
            })
        except:
            pass
    
    return charts if charts else []
    """Create predefined business-focused visualizations"""
    charts = []
    
    # Helper function to find columns by keywords
    def find_column(keywords, columns):
        """Find first column matching any keyword"""
        keywords = [k.lower() for k in keywords]
        for col in columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None
    
    # Get column types
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Find key columns
    date_col = date_cols[0] if date_cols else None
    sales_col = find_column(['sales', 'revenue', 'amount', 'total', 'price'], numeric_cols)
    quantity_col = find_column(['quantity', 'qty', 'units', 'count'], numeric_cols)
    product_col = find_column(['product', 'item', 'productline', 'category'], categorical_cols)
    customer_col = find_column(['customer', 'client', 'customername', 'account'], categorical_cols)
    country_col = find_column(['country', 'region', 'territory', 'location'], categorical_cols)
    
    # Use sales_col as default metric if found
    metric_col = sales_col or numeric_cols[0] if numeric_cols else None
    
    # 1. Daily Sales Trend (if date and sales columns exist)
    if date_col and metric_col:
        try:
            # Aggregate by date
            daily_data = df.groupby(date_col)[metric_col].sum().reset_index()
            
            fig = px.line(
                daily_data,
                x=date_col,
                y=metric_col,
                title=f'Daily {metric_col} Trend',
                markers=True
            )
            fig.update_layout(
                hovermode='x unified',
                height=400,
                xaxis_title='Date',
                yaxis_title=metric_col
            )
            charts.append({
                'figure': fig,
                'title': f'Daily {metric_col} Trend',
                'purpose': 'Track daily performance and identify patterns'
            })
        except Exception as e:
            st.warning(f"Could not create daily trend chart: {str(e)}")
    
    # 2. Sales by Product Line (if product and sales columns exist)
    if product_col and metric_col:
        try:
            product_sales = df.groupby(product_col)[metric_col].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=product_sales.index,
                y=product_sales.values,
                title=f'{metric_col} by {product_col}',
                labels={'x': product_col, 'y': metric_col}
            )
            fig.update_layout(
                height=400,
                xaxis_title=product_col,
                yaxis_title=f'Total {metric_col}',
                showlegend=False
            )
            charts.append({
                'figure': fig,
                'title': f'{metric_col} by {product_col}',
                'purpose': 'Compare performance across product categories'
            })
        except Exception as e:
            st.warning(f"Could not create product chart: {str(e)}")
    
    # 3. Monthly Sales Trend (if date and sales columns exist)
    if date_col and metric_col:
        try:
            # Extract month and aggregate
            df_copy = df.copy()
            df_copy['Month'] = pd.to_datetime(df_copy[date_col]).dt.to_period('M').astype(str)
            monthly_data = df_copy.groupby('Month')[metric_col].sum().reset_index()
            
            fig = px.line(
                monthly_data,
                x='Month',
                y=metric_col,
                title=f'Monthly {metric_col} Trend',
                markers=True
            )
            fig.update_layout(
                height=400,
                xaxis_title='Month',
                yaxis_title=f'Total {metric_col}'
            )
            charts.append({
                'figure': fig,
                'title': f'Monthly {metric_col} Trend',
                'purpose': 'Identify monthly patterns and growth trends'
            })
        except Exception as e:
            st.warning(f"Could not create monthly trend chart: {str(e)}")
    
    # 4. Top 10 Customers (if customer and sales columns exist)
    if customer_col and metric_col:
        try:
            top_customers = df.groupby(customer_col)[metric_col].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=top_customers.values,
                y=top_customers.index,
                orientation='h',
                title=f'Top 10 {customer_col} by {metric_col}',
                labels={'x': f'Total {metric_col}', 'y': customer_col}
            )
            fig.update_layout(
                height=500,
                showlegend=False
            )
            charts.append({
                'figure': fig,
                'title': f'Top 10 {customer_col}',
                'purpose': 'Identify most valuable customers'
            })
        except Exception as e:
            st.warning(f"Could not create top customers chart: {str(e)}")
    
    # 5. Sales by Country/Region (if country and sales columns exist)
    if country_col and metric_col:
        try:
            country_sales = df.groupby(country_col)[metric_col].sum().sort_values(ascending=False).head(15)
            
            fig = px.bar(
                x=country_sales.index,
                y=country_sales.values,
                title=f'{metric_col} by {country_col}',
                labels={'x': country_col, 'y': f'Total {metric_col}'}
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            charts.append({
                'figure': fig,
                'title': f'{metric_col} by {country_col}',
                'purpose': 'Compare regional performance'
            })
        except Exception as e:
            st.warning(f"Could not create country chart: {str(e)}")
    
    # 6. Quantity Analysis (if quantity column exists)
    if quantity_col and product_col:
        try:
            product_qty = df.groupby(product_col)[quantity_col].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=product_qty.index,
                y=product_qty.values,
                title=f'Top Products by {quantity_col}',
                labels={'x': product_col, 'y': f'Total {quantity_col}'}
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            charts.append({
                'figure': fig,
                'title': f'Top Products by {quantity_col}',
                'purpose': 'Identify most popular products by volume'
            })
        except Exception as e:
            st.warning(f"Could not create quantity chart: {str(e)}")
    
    # If no charts were created, add a message
    if not charts:
        st.info("📊 Could not generate standard business charts. Your data may need columns like: date, sales/revenue, product, customer, or country.")
    
    return charts
    """Use LLM to determine the most useful visualizations for the data"""
    
    if not openai_client:
        # Fallback to default charts if no OpenAI
        return create_default_visualizations(df)
    
    try:
        # Prepare data summary for LLM
        data_summary = {
            'columns': df.columns.tolist(),
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'column_types': {
                'numeric': df.select_dtypes(include=['number']).columns.tolist(),
                'categorical': df.select_dtypes(include=['object']).columns.tolist(),
                'datetime': df.select_dtypes(include=['datetime64']).columns.tolist()
            },
            'sample_data': df.head(3).to_dict('records'),
            'insights': [{'title': i['title'], 'content': i['content']} for i in insights[:3]]
        }
        
        prompt = f"""
You are a business intelligence expert analyzing a dataset to recommend visualizations.

Dataset Information:
{json.dumps(data_summary, indent=2, default=str)}

Based on this data and the insights generated, recommend 3 visualizations that would be most valuable for understanding business trends.

For each visualization, specify:
1. chart_type: "line", "bar", "histogram", or "scatter"
2. x_column: exact column name from the data
3. y_column: exact column name (for line/bar/scatter)
4. color_by: optional grouping column name (for line/bar)
5. title: descriptive title
6. business_purpose: why this chart is valuable

Respond ONLY with a JSON array of 3 visualization specifications. Example format:
[
  {{
    "chart_type": "line",
    "x_column": "ORDERDATE",
    "y_column": "SALES",
    "color_by": "COUNTRY",
    "title": "Sales Trends by Country Over Time",
    "business_purpose": "Identify seasonal patterns and country performance"
  }},
  {{
    "chart_type": "bar",
    "x_column": "PRODUCTLINE",
    "y_column": "SALES",
    "title": "Total Sales by Product Line",
    "business_purpose": "Compare product line performance"
  }},
  {{
    "chart_type": "histogram",
    "x_column": "SALES",
    "title": "Distribution of Order Values",
    "business_purpose": "Understand typical order sizes and outliers"
  }}
]

IMPORTANT: Use ONLY column names that exist in the data. Do not invent columns.
"""
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a business intelligence expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        response_text = response_text.strip()
        
        viz_specs = json.loads(response_text)
        
        # Create the recommended visualizations
        charts = []
        for spec in viz_specs[:3]:  # Limit to 3
            try:
                fig = create_chart_from_spec(df, spec)
                if fig:
                    charts.append({
                        'figure': fig,
                        'title': spec.get('title', 'Chart'),
                        'purpose': spec.get('business_purpose', '')
                    })
            except Exception as e:
                st.warning(f"Could not create chart: {spec.get('title', 'Unknown')} - {str(e)}")
                continue
        
        return charts if charts else create_default_visualizations(df)
        
    except Exception as e:
        st.warning(f"AI visualization recommendation failed: {str(e)}. Using default charts.")
        return create_default_visualizations(df)

def create_chart_from_spec(df, spec):
    """Create a chart based on LLM specification"""
    chart_type = spec.get('chart_type', '').lower()
    x_col = spec.get('x_column')
    y_col = spec.get('y_column')
    color_by = spec.get('color_by')
    title = spec.get('title', 'Chart')
    
    # Validate columns exist
    if x_col and x_col not in df.columns:
        return None
    if y_col and y_col not in df.columns:
        return None
    if color_by and color_by not in df.columns:
        color_by = None
    
    try:
        if chart_type == 'line':
            if color_by:
                # Limit to top 10 categories for readability
                top_categories = df[color_by].value_counts().head(10).index
                df_filtered = df[df[color_by].isin(top_categories)]
                fig = px.line(
                    df_filtered.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    color=color_by,
                    title=title,
                    markers=True
                )
            else:
                fig = px.line(
                    df.sort_values(x_col),
                    x=x_col,
                    y=y_col,
                    title=title,
                    markers=True
                )
            fig.update_layout(height=400, hovermode='x unified')
            return fig
            
        elif chart_type == 'bar':
            if y_col:
                # Aggregate data
                if color_by:
                    grouped = df.groupby([x_col, color_by])[y_col].sum().reset_index()
                    fig = px.bar(
                        grouped.nlargest(15, y_col),
                        x=x_col,
                        y=y_col,
                        color=color_by,
                        title=title
                    )
                else:
                    grouped = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(15)
                    fig = px.bar(
                        x=grouped.index,
                        y=grouped.values,
                        title=title,
                        labels={'x': x_col, 'y': y_col}
                    )
            else:
                # Count-based bar chart
                counts = df[x_col].value_counts().head(15)
                fig = px.bar(
                    x=counts.index,
                    y=counts.values,
                    title=title,
                    labels={'x': x_col, 'y': 'Count'}
                )
            fig.update_layout(height=400)
            return fig
            
        elif chart_type == 'histogram':
            fig = px.histogram(
                df,
                x=x_col,
                title=title,
                nbins=30
            )
            fig.update_layout(height=400)
            return fig
            
        elif chart_type == 'scatter':
            if y_col:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color_by if color_by else None,
                    title=title,
                    trendline='ols' if not color_by else None
                )
                fig.update_layout(height=400)
                return fig
        
        return None
        
    except Exception as e:
        return None

def create_default_visualizations(df):
    """Fallback: Create default smart visualizations"""
    charts = []
    
    # Chart 1: Trend chart
    fig1 = create_trend_chart(df, None)
    if fig1:
        charts.append({
            'figure': fig1,
            'title': 'Trend Analysis',
            'purpose': 'Identify patterns over time'
        })
    
    # Chart 2: Comparison chart
    fig2 = create_comparison_chart(df)
    if fig2:
        charts.append({
            'figure': fig2,
            'title': 'Category Comparison',
            'purpose': 'Compare performance across categories'
        })
    
    # Chart 3: Distribution
    fig3 = create_distribution_chart(df)
    if fig3:
        charts.append({
            'figure': fig3,
            'title': 'Value Distribution',
            'purpose': 'Understand data spread and outliers'
        })
    
    return charts
    """Create interactive trend visualization"""
    date_cols = df.select_dtypes(include=['datetime64']).columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(date_cols) > 0 and len(numeric_cols) > 0:
        date_col = date_cols[0]
        
        # Smart metric selection - prefer sales/revenue/amount columns
        priority_metrics = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'total', 'quantity', 'value'])]
        metric_col = priority_metrics[0] if priority_metrics else numeric_cols[0]
        
        # Check for categorical columns for grouping
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Avoid ID columns for grouping
        good_cats = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', 'number', 'code', 'key'])]
        
        if len(good_cats) > 0:
            group_col = good_cats[0]
            
            # Limit to top categories if too many
            top_categories = df[group_col].value_counts().head(10).index
            df_filtered = df[df[group_col].isin(top_categories)]
            
            # Create grouped line chart
            fig = px.line(
                df_filtered.sort_values(date_col),
                x=date_col,
                y=metric_col,
                color=group_col,
                title=f'{metric_col} Trend by {group_col}',
                markers=True
            )
        else:
            fig = px.line(
                df.sort_values(date_col),
                x=date_col,
                y=metric_col,
                title=f'{metric_col} Trend Over Time',
                markers=True
            )
        
        fig.update_layout(
            hovermode='x unified',
            height=400
        )
        return fig
    return None

def create_comparison_chart(df):
    """Create comparison bar chart"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        # Smart column selection
        # Avoid ID-like columns for categories
        good_cats = [c for c in categorical_cols if not any(x in c.lower() for x in ['id', 'number', 'code', 'key'])]
        group_col = good_cats[0] if good_cats else categorical_cols[0]
        
        # Prefer meaningful metrics
        priority_metrics = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'total', 'quantity', 'value'])]
        metric_col = priority_metrics[0] if priority_metrics else numeric_cols[0]
        
        grouped_data = df.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(15)
        
        fig = go.Figure(data=[
            go.Bar(
                x=grouped_data.index,
                y=grouped_data.values,
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title=f'Top 15 {group_col} by {metric_col}',
            xaxis_title=group_col,
            yaxis_title=metric_col,
            height=400
        )
        return fig
    return None

def create_distribution_chart(df):
    """Create distribution histogram"""
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Prefer meaningful columns for distribution
        priority_cols = [c for c in numeric_cols if any(x in c.lower() for x in ['sales', 'revenue', 'amount', 'price', 'quantity', 'value'])]
        metric_col = priority_cols[0] if priority_cols else numeric_cols[0]
        
        fig = px.histogram(
            df,
            x=metric_col,
            title=f'Distribution of {metric_col}',
            nbins=30
        )
        
        fig.update_layout(height=400)
        return fig
    return None

# Main App
def main():
    st.title("📊 AI Data Analyst")
    st.markdown("### Automated Data Insights & Recommendations")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        uploaded_file = st.file_uploader("Upload Dataset", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                # ---- TRY MULTIPLE ENCODINGS ----
                if uploaded_file.name.endswith(".csv"):
                    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                    df = None

                    for enc in encodings:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=enc)
                            # st.success(f"File loaded successfully (encoding: {enc})")
                            break
                        except Exception:
                            pass

                    if df is None:
                        raise ValueError("Unable to read CSV with fallback encodings.")

                else:
                    df = pd.read_excel(uploaded_file)

                # Store for main UI
                st.session_state.df = df

            except Exception as e:
                st.error(f"Failed to load file: {e}")

        # ---- SHOW PREVIEW ----
        if "df" in st.session_state:
            st.divider()
            st.subheader("📋 Data Preview")
            st.dataframe(st.session_state.df.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Rows", len(st.session_state.df))
            col2.metric("Columns", len(st.session_state.df.columns))
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Load data with encoding detection
            if uploaded_file.name.endswith('.csv'):
                # Try multiple encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
                df = None
                
                for encoding in encodings:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        # st.success(f"✅ File loaded successfully (encoding: {encoding})")
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                    except Exception as e:
                        if 'utf-8' not in str(e).lower():
                            raise
                
                if df is None:
                    raise ValueError("Unable to read CSV file. Please ensure it's properly formatted.")
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            # Initialize analyzer
            analyzer = DataAnalyzer(df)
            df = analyzer.preprocess_data()
            
            # Update session state with preprocessed df
            st.session_state.df = df
            
            # Generate insights
            with st.spinner("🔍 Analyzing data..."):
                insights = analyzer.generate_insights()
                recommendations = analyzer.generate_recommendations(insights)
            
            # Display insights
            st.header("🎯 Key Insights")
            
            insight_cols = st.columns(2)
            for idx, insight in enumerate(insights):
                with insight_cols[idx % 2]:
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>{insight['title']}</h4>
                        <p>{insight['content']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.checkbox(f"Show details - {insight['title']}", key=f"insight_{idx}"):
                        st.json(insight['data'])
            
            st.divider()
            
            # Visualizations - AI-Generated (No Hardcoding)
            st.header("📈 Data Visualizations")
            # st.caption("🤖 AI-powered chart generation - adapts to your specific data")
            
            with st.spinner("🎨 Analyzing your data and generating relevant visualizations..."):
                charts = generate_dynamic_visualizations(df, insights, openai_client)
            
            if charts:
                # Create tabs for different charts
                tab_names = [f"{i+1}. {chart['title']}" for i, chart in enumerate(charts)]
                tabs = st.tabs(tab_names)
                
                for tab, chart_info in zip(tabs, charts):
                    with tab:
                        st.markdown(f"**Business Question:** {chart_info['purpose']}")
                        st.plotly_chart(chart_info['figure'], use_container_width=True)
            else:
                st.warning("⚠️ Unable to generate visualizations. Please ensure your data has numeric columns and optionally date/category columns.")
            
            st.divider()
            
            # Recommendations
            st.header("💡 Recommendations")
            
            for idx, rec in enumerate(recommendations):
                priority_color = {
                    'High': '🔴',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(rec['priority'], '⚪')
                
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>{priority_color} {rec['priority']} Priority</h4>
                    <p><strong>Action:</strong> {rec['action']}</p>
                    <p><strong>Reason:</strong> {rec['reason']}</p>
                    <p><strong>Expected Impact:</strong> {rec['expected_impact']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # AI Narrative - always try to generate if OpenAI is available
            if openai_client:
                st.header("📝 Executive Summary")
                
                with st.spinner("Generating AI narrative..."):
                    summary = analyzer.generate_summary_stats()
                    narrative = generate_ai_narrative(insights, recommendations, summary)
                
                st.markdown(f"""
                <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; border-left: 5px solid #ff7f0e;">
                {narrative}
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Download report
            st.header("📥 Export Report")
            
            report_data = {
                'insights': insights,
                'recommendations': recommendations,
                'summary': analyzer.generate_summary_stats()
            }
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download JSON Report",
                    data=safe_json_dumps(report_data),
                    file_name="analysis_report.json",
                    mime="application/json"
                )
            
            with col2:
                # Create CSV summary
                insights_df = pd.DataFrame(insights)
                csv = insights_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Summary",
                    data=csv,
                    file_name="insights_summary.csv",
                    mime="text/csv"
                )
            
            st.divider()
            
            # CHAT INTERFACE - Main Content Area
            st.header("💬 Ask Questions About Your Data")
            
            # Show active chat mode with icon
            # if LANGCHAIN_AVAILABLE and openai_client:
            #     st.success("🚀 **Active Mode:** LangChain Agent - Most powerful analysis with natural language understanding")
            # elif openai_client:
            #     st.info("🤖 **Active Mode:** Enhanced Query - Data analysis with AI explanations")
            # else:
            #     st.info("📊 **Active Mode:** Smart Query - Direct data analysis (no API key needed)")
            
            if not LANGCHAIN_AVAILABLE and openai_client:
                st.warning("💡 Install LangChain for enhanced querying: `pip install langchain langchain-experimental langchain-openai tabulate`")
            elif not openai_client:
                st.warning("💡 Add OpenAI API key to `.env` file for AI-powered features")
            
            # Initialize chat history
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        with st.chat_message(message["role"]):
                            content = message["content"]
                            
                            # Check if content is a plot
                            if isinstance(content, tuple) and len(content) == 2:
                                fig, description = content
                                st.markdown(description)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.markdown(content)
                else:
                    st.info("👋 Start by asking a question below! Examples:\n- Which city ordered the most vintage cars?\n- What is the total revenue?\n- Plot revenue by region\n- Show me a trend chart")
            
            # Chat input
            user_query = st.chat_input(
                "Ask a question about your data...",
                key="main_chat_input"
            )
            
            # Process query
            if user_query:
                # Add user message
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_query
                })
                
                # Show which method is being used
                method_status = "🚀 Using LangChain Agent..." if (LANGCHAIN_AVAILABLE and openai_client) else "📊 Analyzing data..."
                
                # Get answer
                with st.spinner(method_status):
                    answer, error = answer_data_question(df, user_query)
                    
                    if error:
                        error_msg = f"❌ **Error:** {error}\n\n💡 Try rephrasing your question or check that your data contains the information you're asking about."
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                    else:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })
                
                st.rerun()
            
            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen
        st.info("👈 Upload a CSV or Excel file to begin analysis")
        
        st.markdown("""
        ### Features:
        - 📊 **Automated Insights**: AI-powered trend detection and analysis
        - 🔍 **Anomaly Detection**: Identify unusual patterns in your data
        - 📈 **Interactive Visualizations**: Dynamic charts and graphs
        - 💡 **Smart Recommendations**: Actionable business advice
        - 💬 **Chat Interface**: Ask questions in natural language and get instant answers with real data
        - 📊 **On-Demand Plotting**: Request custom charts through chat
        - 📝 **AI Narrative**: Executive summary generation (requires API key)
        
        ### How It Works:
        1. **Upload** your CSV or Excel file
        2. **Review** automatic insights and visualizations
        3. **Chat** with your data - ask any question or request plots!
        4. **Export** reports and insights
        
        ### Example Questions You Can Ask:
        - "Which city ordered the most vintage cars?"
        - "What is the total revenue by region?"
        - "Show me the top 5 products by sales"
        - "Plot revenue over time"
        - "Create a bar chart comparing regions"
        - "Show me a distribution of order values"
        """)
        
        st.info("💡 **Works with or without API key!** Basic features available offline, enhanced with OpenAI.")
        
        # Load sample data option
        if st.button("📂 Load Sample Dataset"):
            sample_path = "sample_data/sales_data.csv"
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
                st.session_state.df = df
                st.rerun()
            else:
                st.warning("Sample dataset not found. Please upload your own file.")

if __name__ == "__main__":
    main()