"""
Monitoring and analytics utilities for the Legal Document Analyzer.
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from src.core.config import settings
from src.utils.logging import app_logger


class MonitoringSystem:
    """System for monitoring query performance and model behavior."""
    
    def __init__(self):
        self.metrics_file = Path(settings.log_file).parent / "metrics.jsonl"
        self.session_metrics = []
    
    def log_query_metrics(self, query_data: Dict[str, Any]) -> None:
        """Log query metrics for analysis."""
        try:
            # Add timestamp
            query_data["timestamp"] = datetime.now().isoformat()
            query_data["unix_timestamp"] = time.time()
            
            # Add to session metrics
            self.session_metrics.append(query_data)
            
            # Write to file
            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(query_data) + '\n')
            
        except Exception as e:
            app_logger.error(f"Error logging metrics: {str(e)}")
    
    def calculate_hallucination_score(self, answer: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate a simple hallucination confidence score."""
        try:
            if not sources or not answer:
                return 0.0
            
            # Extract source content
            source_text = " ".join([s.get("content", "") for s in sources])
            if not source_text:
                return 0.0
            
            # Simple word overlap analysis
            answer_words = set(answer.lower().split())
            source_words = set(source_text.lower().split())
            
            # Remove common words
            common_words = {
                "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", 
                "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would", "could",
                "should", "may", "might", "must", "shall", "can", "this", "that",
                "these", "those", "i", "you", "he", "she", "it", "we", "they"
            }
            
            answer_words_filtered = answer_words - common_words
            source_words_filtered = source_words - common_words
            
            if not answer_words_filtered:
                return 0.5
            
            # Calculate overlap ratio
            overlap = len(answer_words_filtered.intersection(source_words_filtered))
            overlap_ratio = overlap / len(answer_words_filtered)
            
            # Boost score for longer answers (more content = potentially more grounded)
            length_factor = min(len(answer) / 200, 1.0)
            
            # Combine factors
            confidence_score = (overlap_ratio * 0.7) + (length_factor * 0.3)
            
            return min(confidence_score, 1.0)
            
        except Exception as e:
            app_logger.error(f"Error calculating hallucination score: {str(e)}")
            return 0.5
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics for the current session."""
        try:
            if not self.session_metrics:
                return {}
            
            # Calculate basic metrics
            total_queries = len(self.session_metrics)
            confidences = [m.get("confidence", 0) for m in self.session_metrics]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # High confidence queries (above threshold)
            high_conf_queries = len([c for c in confidences if c >= settings.confidence_threshold])
            high_conf_rate = high_conf_queries / total_queries if total_queries > 0 else 0
            
            # Response times (if available)
            response_times = [m.get("response_time", 0) for m in self.session_metrics if m.get("response_time")]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # Most common question types (simple keyword analysis)
            question_keywords = {}
            for metric in self.session_metrics:
                question = metric.get("question", "").lower()
                for word in ["termination", "liability", "payment", "confidentiality", "intellectual", "breach"]:
                    if word in question:
                        question_keywords[word] = question_keywords.get(word, 0) + 1
            
            return {
                "total_queries": total_queries,
                "average_confidence": avg_confidence,
                "high_confidence_rate": high_conf_rate,
                "average_response_time": avg_response_time,
                "question_keywords": question_keywords,
                "confidence_distribution": {
                    "high": len([c for c in confidences if c >= 0.7]),
                    "medium": len([c for c in confidences if 0.4 <= c < 0.7]),
                    "low": len([c for c in confidences if c < 0.4])
                }
            }
            
        except Exception as e:
            app_logger.error(f"Error getting session analytics: {str(e)}")
            return {}
    
    def load_historical_metrics(self) -> List[Dict[str, Any]]:
        """Load historical metrics from file."""
        try:
            if not self.metrics_file.exists():
                return []
            
            metrics = []
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            metrics.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            return metrics
            
        except Exception as e:
            app_logger.error(f"Error loading historical metrics: {str(e)}")
            return []
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analytics report."""
        try:
            # Load all metrics
            all_metrics = self.load_historical_metrics()
            session_analytics = self.get_session_analytics()
            
            if not all_metrics:
                return {"error": "No metrics available"}
            
            # Time-based analysis
            df = pd.DataFrame(all_metrics)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp")
                
                # Daily aggregations
                daily_stats = df.groupby(df["timestamp"].dt.date).agg({
                    "confidence": ["mean", "count"],
                    "sources_count": "mean"
                }).round(3)
            else:
                daily_stats = None
            
            # Overall statistics
            total_queries = len(all_metrics)
            avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0
            
            # Confidence trends
            confidence_over_time = df[["timestamp", "confidence"]].to_dict("records") if "timestamp" in df.columns else []
            
            # Model performance
            if "model" in df.columns:
                model_performance = df.groupby("model")["confidence"].agg(["mean", "count"]).to_dict()
            else:
                model_performance = {}
            
            report = {
                "report_generated": datetime.now().isoformat(),
                "total_queries_all_time": total_queries,
                "average_confidence_all_time": avg_confidence,
                "session_analytics": session_analytics,
                "confidence_trend": confidence_over_time[-50:],  # Last 50 queries
                "model_performance": model_performance,
                "recommendations": self._generate_recommendations(all_metrics)
            }
            
            return report
            
        except Exception as e:
            app_logger.error(f"Error generating analytics report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on metrics analysis."""
        recommendations = []
        
        try:
            if not metrics:
                return ["No data available for recommendations."]
            
            # Calculate average confidence
            confidences = [m.get("confidence", 0) for m in metrics]
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < 0.5:
                recommendations.append(
                    "Low average confidence detected. Consider adding more relevant documents to improve answer quality."
                )
            
            if avg_confidence < 0.3:
                recommendations.append(
                    "Very low confidence scores. Check if uploaded documents are relevant to typical queries."
                )
            
            # Check for low source counts
            source_counts = [m.get("sources_count", 0) for m in metrics]
            avg_sources = sum(source_counts) / len(source_counts) if source_counts else 0
            
            if avg_sources < 2:
                recommendations.append(
                    "Low number of sources being used. Consider adjusting chunk size or search parameters."
                )
            
            # Check query patterns
            recent_metrics = metrics[-20:] if len(metrics) > 20 else metrics
            recent_confidences = [m.get("confidence", 0) for m in recent_metrics]
            recent_avg = sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0
            
            if recent_avg < avg_confidence - 0.1:
                recommendations.append(
                    "Recent query performance has declined. Consider refreshing the knowledge base."
                )
            
            if not recommendations:
                recommendations.append("Performance looks good! Keep monitoring for continued optimal results.")
            
        except Exception as e:
            app_logger.error(f"Error generating recommendations: {str(e)}")
            recommendations.append("Unable to generate recommendations due to data processing error.")
        
        return recommendations


# Global monitoring instance
monitoring_system = MonitoringSystem()
