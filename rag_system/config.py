from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class Settings:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "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")
    
    # Vector DB Settings
    VECTOR_DB_PATH: str = "storage/vector_stores"
    
    # Model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    
    # Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    
    # Retrieval Settings
    TOP_K_RESULTS: int = 4

    # Visualization Settings
    CHART_TYPES = {
        "environmental": ["line", "bar", "scatter"],
        "social": ["pie", "bar", "radar"],
        "governance": ["bar", "heatmap", "treemap"]
    }

    # Analysis Templates
    ANALYSIS_ASPECTS = {
        "Environmental Performance": {
            "prompt": """
            Extract and analyze key environmental metrics and initiatives:
            1. Carbon emissions (Scope 1, 2, 3) and targets
            2. Energy consumption and renewable energy usage
            3. Water management and efficiency
            4. Waste management and recycling rates
            5. Environmental compliance and incidents
            
            Format the response as bullet points with:
            - Specific metrics and numbers where available
            - Year-over-year comparisons if present
            - Targets and progress against them
            """,
            "icon": "🌍"
        },
        "Social Impact": {
            "prompt": """
            Extract and analyze key social metrics and initiatives:
            1. Workforce diversity statistics and targets
            2. Employee training hours and development programs
            3. Health and safety incidents and rates
            4. Community engagement and social investment
            5. Human rights and labor practices
            
            Format the response as bullet points with:
            - Specific metrics and numbers where available
            - Progress on diversity and inclusion
            - Safety performance metrics
            """,
            "icon": "👥"
        },
        "Governance Structure": {
            "prompt": """
            Extract and analyze key governance metrics and practices:
            1. Board composition and diversity
            2. Ethics and compliance programs
            3. Risk management framework
            4. Stakeholder engagement practices
            5. Executive compensation and ESG links
            
            Format the response as bullet points with:
            - Board diversity percentages
            - Number of independent directors
            - Ethics violation statistics
            """,
            "icon": "⚖️"
        }
    }

    # Validation Settings
    REQUIRED_SECTIONS = [
        "environmental",
        "social",
        "governance",
        "metrics",
        "targets"
    ]

    DATA_QUALITY_RULES = {
        "emissions": r"\d+(\.\d+)?\s*(tCO2e|CO2|GHG)",
        "percentages": r"\d+(\.\d+)?%",
        "monetary": r"[\$€£]\s*\d+(\.\d+)?\s*(million|billion|M|B)?"
    }

    # Dashboard Queries
    PREDEFINED_QUERIES = {
        "environmental_metrics": """
        Extract all quantitative environmental metrics including:
        - Carbon emissions
        - Energy consumption
        - Water usage
        - Waste metrics
        Return as structured data.
        """,
        "social_metrics": """
        Extract all quantitative social metrics including:
        - Workforce diversity percentages
        - Safety incident rates
        - Training hours
        - Community investment amounts
        Return as structured data.
        """,
        "governance_metrics": """
        Extract all quantitative governance metrics including:
        - Board diversity
        - Ethics violations
        - Compliance rates
        Return as structured data.
        """
    }

settings = Settings()