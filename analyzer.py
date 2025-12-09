import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json


def convert_to_serializable(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


class DataAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    # ----------------------------------------------------------
    # PREPROCESSING
    # ----------------------------------------------------------
    def preprocess_data(self):
        """Ensure correct dtypes for analysis."""
        # Correct date column
        if "ORDERDATE" in self.df.columns:
            self.df["ORDERDATE"] = pd.to_datetime(self.df["ORDERDATE"], errors="coerce")

        # Strip whitespace in columns
        self.df.columns = self.df.columns.str.strip()

        return self.df

    # ----------------------------------------------------------
    #   BUSINESS-RELEVANT COLUMNS
    # ----------------------------------------------------------
    @property
    def date_col(self):
        return "ORDERDATE"

    @property
    def metric_col(self):
        return "SALES"

    @property
    def primary_group(self):
        """Product Line preferred; else Territory; else Country."""
        for col in ["PRODUCTLINE", "COUNTRY", "CITY"]:
            if col in self.df.columns:
                return col
        return None

    # ----------------------------------------------------------
    # SUMMARY STATISTICS
    # ----------------------------------------------------------
    def generate_summary_stats(self):
        summary = {
            "total_orders": len(self.df),
            "total_sales": float(self.df["SALES"].sum()),
            "avg_order_value": float(self.df["SALES"].mean()),
        }

        if self.date_col in self.df:
            summary["date_range"] = {
                "start": str(self.df[self.date_col].min().date()),
                "end": str(self.df[self.date_col].max().date()),
            }

        return summary

    # ----------------------------------------------------------
    # TOP PERFORMERS
    # ----------------------------------------------------------
    def top_segments(self):
        group = self.primary_group
        metric = self.metric_col

        if group:
            grouped = (
                self.df.groupby(group)[metric]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )

            return [
                {"segment": seg, "sales": float(val)} for seg, val in grouped.items()
            ]

        return []

    # ----------------------------------------------------------
    # TRENDS OVER TIME
    # ----------------------------------------------------------
    def calculate_trends(self):
        """Month-over-Month SALES trends by PRODUCTLINE."""
        if self.date_col not in self.df:
            return []

        df = self.df.copy()

        df["MONTH"] = df[self.date_col].dt.to_period("M")

        group = self.primary_group
        metric = self.metric_col

        trends = []

        if group:
            monthly = df.groupby([group, "MONTH"])[metric].sum().reset_index()

            for product in monthly[group].unique():
                data = monthly[monthly[group] == product].sort_values("MONTH")

                if len(data) < 2:
                    continue

                recent = data.iloc[-1][metric]
                previous = data.iloc[-2][metric]

                if previous > 0:
                    pct_change = ((recent - previous) / previous) * 100

                    trends.append(
                        {
                            "segment": product,
                            "current": float(recent),
                            "previous": float(previous),
                            "pct_change": float(pct_change),
                            "direction": "increase" if pct_change > 0 else "decrease",
                        }
                    )

        return trends

    # ----------------------------------------------------------
    # ANOMALIES
    # ----------------------------------------------------------
    def detect_anomalies(self):
        """Detect anomalies in SALES & QUANTITYORDERED."""
        numeric_cols = [col for col in ["SALES", "QUANTITYORDERED"] if col in self.df]

        if not numeric_cols:
            return []

        model = IsolationForest(contamination=0.05, random_state=42)
        scores = model.fit_predict(self.df[numeric_cols])

        anomalies = self.df[scores == -1].head(5)

        result = []
        for _, row in anomalies.iterrows():
            result.append(
                {col: convert_to_serializable(row[col]) for col in self.df.columns}
            )

        return result

    # ----------------------------------------------------------
    # FULL INSIGHT GENERATION
    # ----------------------------------------------------------
    def generate_insights(self):
        insights = []

        # SUMMARY
        summary = self.generate_summary_stats()
        insights.append(
            {
                "type": "summary",
                "title": "Dataset Summary",
                "content": f"Processed {summary['total_orders']} orders totaling ${summary['total_sales']:,}.",
                "data": summary,
            }
        )

        # TOP SEGMENTS
        top = self.top_segments()
        if top:
            insights.append(
                {
                    "type": "top_performers",
                    "title": "Top Revenue-Generating Segments",
                    "content": f"Highest revenue segment: {top[0]['segment']} (${top[0]['sales']:,}).",
                    "data": top,
                }
            )

        # TRENDS
        trends = self.calculate_trends()
        if trends:
            most_sig = max(trends, key=lambda t: abs(t["pct_change"]))
            insights.append(
                {
                    "type": "trend",
                    "title": "Key Sales Trend",
                    "content": f"{most_sig['segment']} saw a {most_sig['pct_change']:.1f}% {most_sig['direction']} in monthly sales.",
                    "data": most_sig,
                }
            )

        # ANOMALIES
        anomalies = self.detect_anomalies()
        if anomalies:
            insights.append(
                {
                    "type": "anomaly",
                    "title": "Outlier Orders Detected",
                    "content": "Detected unusual sales or quantity patterns requiring review.",
                    "data": anomalies,
                }
            )

        return insights

    # ----------------------------------------------------------
    # RECOMMENDATIONS
    # ----------------------------------------------------------
    def generate_recommendations(self, insights):
        recs = []

        for ins in insights:
            if ins["type"] == "trend":
                pct = ins["data"]["pct_change"]
                segment = ins["data"]["segment"]

                if pct < -5:
                    recs.append(
                        {
                            "priority": "High",
                            "action": f"Investigate declining sales in {segment}.",
                            "reason": f"Sales dropped {abs(pct):.1f}%.",
                            "expected_impact": "Recover revenue loss",
                        }
                    )
                else:
                    recs.append(
                        {
                            "priority": "Medium",
                            "action": f"Scale successful tactics in {segment}.",
                            "reason": f"Sales rose {pct:.1f}%.",
                            "expected_impact": "Grow revenue",
                        }
                    )

            if ins["type"] == "anomaly":
                recs.append(
                    {
                        "priority": "High",
                        "action": "Review flagged anomalous orders.",
                        "reason": "Unusual SALES or QUANTITYORDERED detected.",
                        "expected_impact": "Reduce operational risk",
                    }
                )

        if not recs:
            recs.append(
                {
                    "priority": "Low",
                    "action": "Monitor monthly sales regularly.",
                    "reason": "No critical issues found.",
                    "expected_impact": "Maintain performance stability",
                }
            )

        return recs[:3]
