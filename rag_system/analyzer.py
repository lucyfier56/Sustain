from typing import Optional, Dict, Any
from core import RAGCore, DocumentProcessor, DashboardEngine
from config import settings
import re
from datetime import datetime

class ReportAnalyzer:
    def __init__(self):
        self.required_sections = settings.REQUIRED_SECTIONS
        self.data_quality_rules = settings.DATA_QUALITY_RULES

    def validate_document(self, text: str) -> Dict[str, Any]:
        validation_results = {
            "completeness": self._check_completeness(text),
            "data_quality": self._check_data_quality(text),
            "consistency": self._check_consistency(text),
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate overall scores
        validation_results["scores"] = self._calculate_scores(validation_results)
        return validation_results

    def _check_completeness(self, text: str) -> Dict[str, bool]:
        results = {}
        for section in self.required_sections:
            # Check for section presence using various forms
            section_patterns = [
                section.lower(),
                section.upper(),
                section.title(),
                f"{section} section",
                f"{section} performance"
            ]
            results[section] = any(pattern in text.lower() for pattern in section_patterns)
        return results

    def _check_data_quality(self, text: str) -> Dict[str, Dict[str, Any]]:
        results = {}
        for metric_type, pattern in self.data_quality_rules.items():
            matches = re.findall(pattern, text)
            results[metric_type] = {
                "count": len(matches),
                "examples": matches[:3],  # Store first 3 examples
                "has_data": len(matches) > 0
            }
        return results

    def _check_consistency(self, text: str) -> Dict[str, bool]:
        return {
            "year_coverage": self._check_year_coverage(text),
            "metric_consistency": self._check_metric_consistency(text),
            "data_formatting": self._check_data_formatting(text)
        }

    def _check_year_coverage(self, text: str) -> bool:
        years = re.findall(r'20\d{2}', text)
        unique_years = sorted(set(years))
        
        if len(unique_years) < 2:
            return False
            
        # Check if years are consecutive
        for i in range(len(unique_years) - 1):
            if int(unique_years[i + 1]) - int(unique_years[i]) > 2:
                return False
        return True

    def _check_metric_consistency(self, text: str) -> bool:
        # Check if metrics are consistently reported across years
        metric_patterns = {
            "emissions": r'(?:emissions|GHG|carbon).*?(\d{4}).*?(\d+(?:\.\d+)?)',
            "energy": r'(?:energy consumption|energy use).*?(\d{4}).*?(\d+(?:\.\d+)?)',
            "water": r'(?:water usage|water consumption).*?(\d{4}).*?(\d+(?:\.\d+)?)'
        }
        
        for pattern in metric_patterns.values():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches or len(matches) < 2:
                return False
        return True

    def _check_data_formatting(self, text: str) -> bool:
        # Check if data is consistently formatted
        formatting_patterns = [
            r'\d+(?:\.\d+)?\s*%',  # Percentages
            r'[\$€£]\s*\d+(?:\.\d+)?(?:\s*[MBK])?',  # Currency
            r'\d+(?:\.\d+)?\s*(?:tCO2e|MWh|m³)'  # Units
        ]
        
        return all(re.search(pattern, text) for pattern in formatting_patterns)

    def _calculate_scores(self, results: Dict[str, Any]) -> Dict[str, float]:
        scores = {}
        
        # Completeness score
        completeness_values = results["completeness"].values()
        scores["completeness"] = sum(completeness_values) / len(completeness_values) * 100
        
        # Data quality score
        quality_values = [item["has_data"] for item in results["data_quality"].values()]
        scores["data_quality"] = sum(quality_values) / len(quality_values) * 100
        
        # Consistency score
        consistency_values = results["consistency"].values()
        scores["consistency"] = sum(consistency_values) / len(consistency_values) * 100
        
        # Overall score
        scores["overall"] = (scores["completeness"] + scores["data_quality"] + scores["consistency"]) / 3
        
        return {k: round(v, 2) for k, v in scores.items()}

class ESGAnalyzer:
    def __init__(self, groq_api_key: str):
        self.rag_core = RAGCore(groq_api_key)
        self.doc_processor = DocumentProcessor()
        self.analysis_aspects = settings.ANALYSIS_ASPECTS
        self.dashboard_engine = DashboardEngine()
        self.report_analyzer = ReportAnalyzer()

    def process_document(self, file_content: bytes, is_private: bool = False) -> Optional[Dict[str, Any]]:
        text = self.doc_processor.process_pdf(file_content)
        if not text:
            return None
            
        store_name = f"esg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analysis_result = self.rag_core.analyze_document(text, store_name, is_private)
        
        if analysis_result:
            # Add validation results
            analysis_result["validation"] = self.report_analyzer.validate_document(text)
            
            # Add initial dashboard data
            analysis_result["dashboard_data"] = self.generate_dashboard_data(
                analysis_result["qa_chain"]
            )
            
        return analysis_result

    def analyze_aspect(self, qa_chain, aspect_name: str) -> Optional[str]:
        aspect = self.analysis_aspects.get(aspect_name)
        if not aspect:
            return None
        return self.rag_core.query(qa_chain, aspect["prompt"])

    def generate_dashboard_data(self, qa_chain) -> Dict[str, Any]:
        dashboard_data = {}
        for metric_type in ["environmental", "social", "governance"]:
            dashboard_data[metric_type] = self.dashboard_engine.generate_visualization_data(
                qa_chain, metric_type
            )
        return dashboard_data

    def generate_summary(self, qa_chain) -> Optional[str]:
        summary_prompt = """
        Based on the sustainability report, provide:
        
        1. Top 3 ESG Strengths:
        - List the most significant achievements
        - Include specific metrics where available
        
        2. Top 3 Areas for Improvement:
        - Identify key gaps or challenges
        - Compare with industry best practices
        
        3. Strategic Recommendations:
        - Provide 3 specific, actionable recommendations
        - Prioritize based on impact and feasibility
        """
        return self.rag_core.query(qa_chain, summary_prompt)

    def custom_query(self, qa_chain, question: str) -> Optional[str]:
        return self.rag_core.query(qa_chain, question)