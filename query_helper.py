"""
Smart Data Query Helper
Automatically analyzes data and provides specific answers to questions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

class DataQueryHelper:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def find_column(self, keyword: str) -> str:
        """Find column name that matches keyword"""
        keyword = keyword.lower()
        for col in self.df.columns:
            if keyword in col.lower():
                return col
        return None
    
    def analyze_top_by_category(self, category_col: str, metric_col: str, top_n: int = 5) -> pd.Series:
        """Get top N categories by metric"""
        return self.df.groupby(category_col)[metric_col].sum().sort_values(ascending=False).head(top_n)
    
    def answer_which_question(self, question: str) -> Dict[str, Any]:
        """Answer 'which' questions like 'which city ordered most'"""
        question_lower = question.lower()
        
        # Find relevant columns from question
        potential_category = None
        potential_metric = None
        
        # Look for category (city, region, product, etc.)
        for col in self.categorical_cols:
            if col.lower() in question_lower or any(word in col.lower() for word in ['city', 'region', 'country', 'product', 'customer']):
                potential_category = col
                break
        
        # Look for metric (quantity, revenue, sales, etc.)
        for col in self.numeric_cols:
            if col.lower() in question_lower or any(word in col.lower() for word in ['quantity', 'amount', 'sales', 'revenue', 'price', 'total']):
                potential_metric = col
                break
        
        if not potential_category:
            potential_category = self.categorical_cols[0] if self.categorical_cols else None
        
        if not potential_metric:
            potential_metric = self.numeric_cols[0] if self.numeric_cols else None
        
        if not potential_category or not potential_metric:
            return None
        
        # Check for specific value filters in question (e.g., "vintage cars")
        filter_value = None
        for col in self.categorical_cols:
            unique_values = self.df[col].unique()
            for val in unique_values:
                if str(val).lower() in question_lower:
                    filter_value = (col, val)
                    break
            if filter_value:
                break
        
        # Apply filter if found
        df_filtered = self.df
        if filter_value:
            filter_col, filter_val = filter_value
            df_filtered = self.df[self.df[filter_col] == filter_val]
        
        # Get top results
        if len(df_filtered) > 0:
            grouped = df_filtered.groupby(potential_category)[potential_metric].sum().sort_values(ascending=False)
            
            result = {
                'type': 'ranking',
                'category': potential_category,
                'metric': potential_metric,
                'filter': filter_value,
                'top_results': grouped.head(5).to_dict(),
                'winner': grouped.index[0] if len(grouped) > 0 else None,
                'winner_value': grouped.iloc[0] if len(grouped) > 0 else None,
                'total_count': len(grouped)
            }
            return result
        
        return None
    
    def answer_what_question(self, question: str) -> Dict[str, Any]:
        """Answer 'what' questions like 'what is total revenue'"""
        question_lower = question.lower()
        
        # Look for aggregation type
        agg_type = 'sum'
        if 'average' in question_lower or 'mean' in question_lower:
            agg_type = 'mean'
        elif 'maximum' in question_lower or 'max' in question_lower or 'highest' in question_lower:
            agg_type = 'max'
        elif 'minimum' in question_lower or 'min' in question_lower or 'lowest' in question_lower:
            agg_type = 'min'
        
        # Find relevant metric
        for col in self.numeric_cols:
            if col.lower() in question_lower:
                value = None
                if agg_type == 'sum':
                    value = self.df[col].sum()
                elif agg_type == 'mean':
                    value = self.df[col].mean()
                elif agg_type == 'max':
                    value = self.df[col].max()
                elif agg_type == 'min':
                    value = self.df[col].min()
                
                return {
                    'type': 'aggregation',
                    'metric': col,
                    'aggregation': agg_type,
                    'value': value
                }
        
        return None
    
    def answer_how_many_question(self, question: str) -> Dict[str, Any]:
        """Answer 'how many' questions"""
        question_lower = question.lower()
        
        # Check if asking about distinct values
        for col in self.categorical_cols:
            if col.lower() in question_lower:
                return {
                    'type': 'count',
                    'category': col,
                    'count': self.df[col].nunique(),
                    'values': self.df[col].unique().tolist()[:10]  # First 10
                }
        
        # General row count
        return {
            'type': 'count',
            'category': 'rows',
            'count': len(self.df)
        }
    
    def query(self, question: str) -> Dict[str, Any]:
        """Main query method - routes to appropriate handler"""
        question_lower = question.lower()
        
        if question_lower.startswith('which') or 'which' in question_lower:
            return self.answer_which_question(question)
        elif question_lower.startswith('what'):
            return self.answer_what_question(question)
        elif 'how many' in question_lower:
            return self.answer_how_many_question(question)
        
        # Default: try which question
        return self.answer_which_question(question)
    
    def format_result(self, result: Dict[str, Any]) -> str:
        """Format query result as readable text"""
        if not result:
            return "I couldn't determine a specific answer from the data."
        
        if result['type'] == 'ranking':
            output = f"**Answer:**\n\n"
            
            if result['filter']:
                filter_col, filter_val = result['filter']
                output += f"Analyzing **{filter_val}** in {filter_col}:\n\n"
            
            output += f"**Top {result['category']} by {result['metric']}:**\n\n"
            
            for i, (cat, val) in enumerate(result['top_results'].items(), 1):
                if i == 1:
                    output += f"🏆 **#{i}: {cat}** - **{val:,.2f}** {result['metric']}\n"
                else:
                    output += f"#{i}: {cat} - {val:,.2f} {result['metric']}\n"
            
            output += f"\n**Winner:** {result['winner']} with {result['winner_value']:,.2f} {result['metric']}"
            
            return output
        
        elif result['type'] == 'aggregation':
            return f"**Answer:** The {result['aggregation']} {result['metric']} is **{result['value']:,.2f}**"
        
        elif result['type'] == 'count':
            if result['category'] == 'rows':
                return f"**Answer:** There are **{result['count']:,}** rows in the dataset."
            else:
                return f"**Answer:** There are **{result['count']}** unique {result['category']} values."
        
        return "Analysis complete."