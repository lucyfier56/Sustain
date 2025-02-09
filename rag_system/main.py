import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import io
import os
import re
from datetime import datetime
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('esg_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


class ESGConfig:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    VECTOR_DB_PATH: str = "storage/vector_stores"
    GROQ_API_KEY: str = "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"
    
    @staticmethod
    def get_chart_colors():
        return {
            'environmental': ['#2ecc71', '#27ae60', '#1abc9c', '#16a085'],
            'social': ['#3498db', '#2980b9', '#9b59b6', '#8e44ad'],
            'governance': ['#f1c40f', '#f39c12', '#e67e22', '#d35400']
        }

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ESGConfig.CHUNK_SIZE,
            chunk_overlap=ESGConfig.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def process_pdf(self, file_content: bytes) -> Optional[str]:
        try:
            pdf_stream = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\nPage {page_num + 1}:\n{page_text}"
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            return text if text.strip() else None
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return None

    def split_text(self, text: str) -> List[Dict[str, str]]:
        chunks = self.text_splitter.split_text(text)
        return [
            {"content": chunk, "source": f"chunk_{i}"} 
            for i, chunk in enumerate(chunks)
        ]

class VectorStoreManager:
    def __init__(self, store_name: str, is_private: bool = False):
        self.store_name = store_name
        self.is_private = is_private
        self.embeddings = HuggingFaceEmbeddings(
            model_name=ESGConfig.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.store_path = os.path.join(
            ESGConfig.VECTOR_DB_PATH,
            "private" if is_private else "public",
            store_name
        )

    def create_store(self, texts: List[Dict[str, str]]) -> FAISS:
        vectorstore = FAISS.from_texts(
            texts=[t["content"] for t in texts],
            embedding=self.embeddings,
            metadatas=texts
        )
        
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        vectorstore.save_local(self.store_path)
        return vectorstore

    def load_store(self) -> Optional[FAISS]:
        if os.path.exists(self.store_path):
            return FAISS.load_local(
                self.store_path,
                self.embeddings
            )
        return None
    
class ESGAnalyzer:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=ESGConfig.LLM_MODEL,
            max_tokens=4096,
            api_key=groq_api_key
        )
        self.doc_processor = DocumentProcessor()

    def create_qa_chain(self, vectorstore: FAISS) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 4}
            ),
            return_source_documents=False
        )

    def process_document(self, file_content: bytes, is_private: bool = False) -> Optional[Dict[str, Any]]:
        text = self.doc_processor.process_pdf(file_content)
        if not text:
            return None
            
        store_name = f"esg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        chunks = self.doc_processor.split_text(text)
        vector_manager = VectorStoreManager(store_name, is_private)
        vectorstore = vector_manager.create_store(chunks)
        
        return {
            "qa_chain": self.create_qa_chain(vectorstore),
            "text": text,
            "vectorstore": vectorstore
        }

class ESGDataFormatter:
    @staticmethod
    def format_for_visualization(qa_chain, raw_response: Union[str, Dict]) -> Optional[Dict[str, Any]]:
        logger.info("Starting data formatting process...")
        
        # Extract the analysis text from RAG response
        try:
            if isinstance(raw_response, dict):
                analysis_text = raw_response.get("result", "")
                logger.info("Extracted analysis text from dictionary response")
            else:
                try:
                    response_dict = json.loads(raw_response)
                    analysis_text = response_dict.get("result", raw_response)
                    logger.info("Extracted analysis text from JSON string")
                except json.JSONDecodeError:
                    analysis_text = raw_response
                    logger.info("Using raw response as analysis text")

            logger.info(f"Analysis text length: {len(analysis_text)}")
            logger.debug("First 200 characters of analysis: " + analysis_text[:200])

        except Exception as e:
            logger.error(f"Error extracting analysis text: {str(e)}")
            return None

        # Create structuring prompt for LLM
        structuring_prompt = f"""
        Analyze this ESG report data and convert it into a structured JSON format.

        Input Data:
        {analysis_text}

        Required JSON Structure:
        {{
            "environmental": {{
                "metrics": [
                    {{
                        "id": "env_emissions",
                        "title": "Carbon Emissions",
                        "visualization": {{
                            "type": "bar",
                            "data": {{
                                "x": ["Scope 1", "Scope 2", "Scope 3"],
                                "y": [actual_values],
                                "units": "tCO2e"
                            }},
                            "properties": {{
                                "x_title": "Emission Scope",
                                "y_title": "Value",
                                "show_legend": true
                            }}
                        }},
                        "insights": "Key findings about this metric"
                    }}
                ],
                "kpis": [
                    {{
                        "title": "Total Emissions",
                        "value": "actual_value tCO2e",
                        "trend": "positive/negative/neutral",
                        "comparison": "vs previous year"
                    }}
                ]
            }},
            "social": {{
                "metrics": [],
                "kpis": []
            }},
            "governance": {{
                "metrics": [],
                "kpis": []
            }}
        }}

        Rules:
        1. Extract ALL numerical values with their units
        2. Use appropriate chart types:
        - line: for time series data
        - bar: for comparisons
        - pie: for percentages
        3. Include ONLY metrics with actual numerical values
        4. Always include units
        5. Add insights for each metric
        6. Ensure chronological ordering for time series

        Return ONLY the JSON structure, no additional text.
        """

        logger.info("Sending structuring prompt to LLM...")
        try:
            # Send to LLM for structuring
            structured_response = qa_chain.invoke({"query": structuring_prompt})
            logger.info("Received response from LLM")
            logger.debug(f"LLM Response type: {type(structured_response)}")
            logger.debug(f"LLM Response: {structured_response}")

            # Extract the JSON part
            try:
                if isinstance(structured_response, dict):
                    response_content = structured_response.get("result", structured_response)
                    logger.info("Extracted result from dictionary response")
                else:
                    response_content = str(structured_response)
                    logger.info("Using string response directly")

                # Clean the response
                logger.info("Cleaning JSON response...")
                json_str = ESGDataFormatter._clean_json_response(response_content)
                logger.debug(f"Cleaned JSON string: {json_str}")

                # Parse and validate
                try:
                    structured_data = json.loads(json_str)
                    logger.info("Successfully parsed JSON")

                    if ESGDataFormatter._validate_structure(structured_data):
                        logger.info("Successfully validated data structure")
                        return structured_data
                    else:
                        logger.warning("Invalid data structure, falling back to basic structure")
                        return ESGDataFormatter._create_fallback_structure(analysis_text)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    logger.error(f"Problematic JSON string: {json_str}")
                    return ESGDataFormatter._create_fallback_structure(analysis_text)

            except Exception as e:
                logger.error(f"Error processing LLM response: {str(e)}")
                return ESGDataFormatter._create_fallback_structure(analysis_text)

        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")
            return ESGDataFormatter._create_fallback_structure(analysis_text)
        
    @staticmethod
    def _debug_llm_response(response: Any) -> None:
        """Helper method to debug LLM response"""
        logger.debug("=== LLM Response Debug ===")
        logger.debug(f"Response type: {type(response)}")
        logger.debug(f"Response content: {response}")
        
        if isinstance(response, dict):
            logger.debug("Dictionary keys: " + str(response.keys()))
            if "result" in response:
                logger.debug("Result content: " + str(response["result"]))

    @staticmethod
    def _clean_json_response(response: str) -> str:
        logger.info("Starting JSON response cleaning")
        try:
            # Try to find the complete JSON structure
            matches = re.findall(r'\{(?:[^{}]|(?R))*\}', response, re.DOTALL)
            if matches:
                for match in matches:
                    try:
                        # Verify it's valid JSON
                        json.loads(match)
                        logger.info("Found valid JSON structure")
                        return match
                    except json.JSONDecodeError:
                        continue

            # If no valid JSON found, try to find the largest JSON-like structure
            start_idx = response.find('{')
            end_idx = response.rindex('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                logger.info("Found JSON structure between curly braces")
                return json_str

        except Exception as e:
            logger.warning(f"Error in primary JSON cleaning: {str(e)}")

        # Fallback: Try to extract JSON from markdown or text
        logger.info("Attempting fallback JSON extraction")
        lines = response.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0
        
        for line in lines:
            if '{' in line:
                in_json = True
            
            if in_json:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    in_json = False
        
        cleaned = '\n'.join(json_lines)
        logger.info(f"Cleaned JSON length: {len(cleaned)}")
        return cleaned

    @staticmethod
    def _validate_structure(data: Dict) -> bool:
        logger.info("Validating data structure")
        try:
            required_categories = ['environmental', 'social', 'governance']
            required_keys = ['metrics', 'kpis']
            
            # Check basic structure
            if not all(category in data for category in required_categories):
                logger.warning("Missing required categories")
                return False
            
            # Check each category
            for category in required_categories:
                if not all(key in data[category] for key in required_keys):
                    logger.warning(f"Missing required keys in {category}")
                    return False
                
                # Validate metrics
                for metric in data[category].get('metrics', []):
                    if not all(key in metric for key in ['id', 'title', 'visualization']):
                        logger.warning(f"Missing required metric keys in {category}")
                        return False
                    if not all(key in metric['visualization'] for key in ['type', 'data', 'properties']):
                        logger.warning(f"Missing required visualization keys in {category}")
                        return False
            
            logger.info("Data structure validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    @staticmethod
    def _create_fallback_structure(analysis_text: str) -> Dict[str, Any]:
        logger.info("Creating fallback structure from raw analysis")
        
        structure = {
            "environmental": {"metrics": [], "kpis": []},
            "social": {"metrics": [], "kpis": []},
            "governance": {"metrics": [], "kpis": []}
        }

        # Define patterns for different types of data
        patterns = {
            'emissions': r'(\d+(?:,\d+)*(?:\.\d+)?)\s*tCO2e',
            'energy': r'(\d+(?:,\d+)*(?:\.\d+)?)\s*MWh',
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'monetary': r'\$?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:million|billion|M|B)?',
            'year_value': r'(\d{4}):\s*(\d+(?:,\d+)*(?:\.\d+)?)'
        }

        # Process each section
        sections = {
            "environmental": r"Environmental[^#]*(?=#|\Z)",
            "social": r"Social[^#]*(?=#|\Z)",
            "governance": r"Governance[^#]*(?=#|\Z)"
        }

        for category, pattern in sections.items():
            match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
            if match:
                section_text = match.group(0)
                logger.info(f"Processing {category} section")
                
                # Extract metrics
                ESGDataFormatter._extract_metrics_from_section(
                    section_text, 
                    category, 
                    patterns, 
                    structure
                )

        logger.info("Fallback structure created")
        return structure

    @staticmethod
    def _extract_metrics_from_section(section_text: str, category: str, patterns: Dict, structure: Dict):
        # Extract time series data
        year_values = re.findall(patterns['year_value'], section_text)
        if year_values:
            structure[category]["metrics"].append({
                "id": f"{category}_timeseries",
                "title": f"{category.title()} Trends",
                "visualization": {
                    "type": "line",
                    "data": {
                        "x": [year for year, _ in year_values],
                        "y": [float(value.replace(',', '')) for _, value in year_values],
                        "units": ESGDataFormatter._detect_units(section_text)
                    },
                    "properties": {
                        "x_title": "Year",
                        "y_title": "Value",
                        "show_legend": True
                    }
                },
                "insights": f"Time series data from {year_values[0][0]} to {year_values[-1][0]}"
            })

        # Extract other numerical values
        for metric_type, pattern in patterns.items():
            if metric_type != 'year_value':
                matches = re.findall(pattern, section_text)
                if matches:
                    structure[category]["kpis"].append({
                        "title": f"{category.title()} {metric_type.title()}",
                        "value": f"{matches[0]} {ESGDataFormatter._detect_units(section_text)}",
                        "trend": "neutral"
                    })

    @staticmethod
    def _detect_units(text: str) -> str:
        unit_patterns = {
            'tCO2e': r'tCO2e',
            'MWh': r'MWh',
            '%': r'%',
            'USD': r'USD|\$',
            'hours': r'hours?'
        }

        for unit, pattern in unit_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return unit
        return ""
        
class ChartCreator:
    def __init__(self):
        self.color_schemes = ESGConfig.get_chart_colors()

    def create_chart(self, metric_data: Dict[str, Any], category: str) -> Optional[go.Figure]:
        try:
            viz_config = metric_data["visualization"]
            chart_type = viz_config["type"]
            data = viz_config["data"]
            props = viz_config["properties"]

            if chart_type == "line":
                return self._create_line_chart(data, props, metric_data["title"], category)
            elif chart_type == "bar":
                return self._create_bar_chart(data, props, metric_data["title"], category)
            elif chart_type == "stacked_bar":
                return self._create_stacked_bar_chart(data, props, metric_data["title"], category)
            elif chart_type == "pie":
                return self._create_pie_chart(data, props, metric_data["title"], category)
            
        except Exception as e:
            print(f"Error creating chart for {metric_data.get('id', 'unknown')}: {str(e)}")
            return None

    def _create_line_chart(self, data, props, title, category):
        fig = go.Figure()

        # Handle single or multiple series
        if "series" in data:
            for idx, series_name in enumerate(data["series"]):
                fig.add_trace(go.Scatter(
                    x=data["x"],
                    y=data["y"][idx] if isinstance(data["y"][0], list) else data["y"],
                    name=series_name,
                    mode='lines+markers',
                    line=dict(
                        color=self.color_schemes[category][idx % len(self.color_schemes[category])],
                        width=2
                    ),
                    marker=dict(size=8)
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data["x"],
                y=data["y"],
                mode='lines+markers',
                line=dict(
                    color=self.color_schemes[category][0],
                    width=2
                ),
                marker=dict(size=8)
            ))

        self._update_layout(fig, title, props, data.get("units"))
        return fig

    def _create_bar_chart(self, data, props, title, category):
        fig = go.Figure()

        if "series" in data:
            for idx, series_name in enumerate(data["series"]):
                fig.add_trace(go.Bar(
                    x=data["x"],
                    y=data["y"][idx] if isinstance(data["y"][0], list) else data["y"],
                    name=series_name,
                    marker_color=self.color_schemes[category][idx % len(self.color_schemes[category])]
                ))
        else:
            fig.add_trace(go.Bar(
                x=data["x"],
                y=data["y"],
                marker_color=self.color_schemes[category][0]
            ))

        if props.get("stacked", False):
            fig.update_layout(barmode='stack')

        self._update_layout(fig, title, props, data.get("units"))
        return fig

    def _create_stacked_bar_chart(self, data, props, title, category):
        fig = go.Figure()

        for idx, series_name in enumerate(data["series"]):
            fig.add_trace(go.Bar(
                x=data["x"],
                y=data["y"][idx],
                name=series_name,
                marker_color=self.color_schemes[category][idx % len(self.color_schemes[category])]
            ))

        fig.update_layout(barmode='stack')
        self._update_layout(fig, title, props, data.get("units"))
        return fig

    def _create_pie_chart(self, data, props, title, category):
        fig = go.Figure(data=[go.Pie(
            labels=data["x"],
            values=data["y"],
            hole=0.3,
            marker=dict(colors=self.color_schemes[category])
        )])

        self._update_layout(fig, title, props)
        return fig

    def _update_layout(self, fig, title, props, units=None):
        # Update y-axis title with units if provided
        y_title = props.get("y_title", "")
        if units and units not in y_title:
            y_title = f"{y_title} ({units})"

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title=props.get("x_title"),
            yaxis_title=y_title,
            showlegend=props.get("show_legend", True),
            height=450,
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)'
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0,0,0,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)'
        )

class DashboardCreator:
    def __init__(self):
        self.chart_creator = ChartCreator()

    def create_category_dashboard(self, category_data: Dict[str, Any], category: str):
        if not category_data:
            st.info(f"No {category.title()} metrics available")
            return

        # Display KPIs if available
        if category_data.get("kpis"):
            self._display_kpis(category_data["kpis"])

        # Display metrics and charts
        if category_data.get("metrics"):
            metrics = category_data["metrics"]
            
            # Create columns for metrics
            cols = st.columns(2)
            for idx, metric in enumerate(metrics):
                with cols[idx % 2]:
                    with st.expander(metric["title"], expanded=True):
                        fig = self.chart_creator.create_chart(metric, category)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            if "insights" in metric:
                                st.info(metric["insights"])

    def _display_kpis(self, kpis: List[Dict[str, Any]]):
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            with col:
                delta_color = "normal"
                if kpi.get("trend") == "positive":
                    delta_color = "good"
                elif kpi.get("trend") == "negative":
                    delta_color = "bad"

                st.metric(
                    label=kpi["title"],
                    value=kpi["value"],
                    delta=kpi.get("comparison"),
                    delta_color=delta_color
                )

    def create_summary_section(self, data: Dict[str, Any]):
        """Create a summary section with key metrics across categories"""
        st.subheader("📊 Key Metrics Summary")
        
        summary_metrics = []
        for category in ["environmental", "social", "governance"]:
            if category in data and "metrics" in data[category]:
                for metric in data[category]["metrics"]:
                    if "data" in metric.get("visualization", {}):
                        viz_data = metric["visualization"]["data"]
                        if isinstance(viz_data.get("y"), list):
                            latest_value = viz_data["y"][-1]
                            summary_metrics.append({
                                "title": metric["title"],
                                "value": f"{latest_value:,.0f}",
                                "unit": viz_data.get("units", ""),
                                "category": category
                            })

        if summary_metrics:
            cols = st.columns(min(3, len(summary_metrics)))
            for idx, metric in enumerate(summary_metrics):
                with cols[idx % 3]:
                    st.metric(
                        label=metric["title"],
                        value=f"{metric['value']} {metric['unit']}"
                    )

class ESGDashboardApp:
    def __init__(self):
        self.dashboard_creator = DashboardCreator()
        self.data_formatter = ESGDataFormatter()
        
    def run(self):
        self._setup_page()
        self._setup_sidebar()
        analyzer = ESGAnalyzer(ESGConfig.GROQ_API_KEY)
        uploaded_file = st.file_uploader("Upload Sustainability Report (PDF)", type="pdf")
        
        if uploaded_file:
            self._process_document(uploaded_file, analyzer)

    def _setup_page(self):
        st.set_page_config(
            page_title="ESG Report Analyzer",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 ESG Report Analysis Dashboard")
        st.markdown("""
        Upload a sustainability report (PDF) to analyze its ESG metrics, initiatives, and commitments.
        This tool provides detailed visualization and analysis of Environmental, Social, and Governance metrics.
        """)

    def _setup_sidebar(self):
        with st.sidebar:
            st.header("Analysis Settings")
            
            st.session_state['private_mode'] = st.checkbox(
                "Enable Private Document Mode", 
                help="Enable for private document analysis with enhanced security"
            )
            
            st.session_state['show_yoy'] = st.checkbox(
                "Include Year-over-Year Analysis",
                value=True,
                help="Compare metrics across different years"
            )
            
            st.session_state['show_insights'] = st.checkbox(
                "Show Data Insights",
                value=True,
                help="Display AI-generated insights for each metric"
            )
            
            if st.button("Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")

    def _process_document(self, uploaded_file, analyzer):
        st.info(f"Processing: {uploaded_file.name}")
        
        with st.spinner("Analyzing document..."):
            analysis_result = analyzer.process_document(
                uploaded_file.read(),
                is_private=st.session_state.get('private_mode', False)
            )
            
            if analysis_result:
                self._display_analysis(analysis_result)
            else:
                st.error("Failed to process the document. Please check the file and try again.")

    def _display_analysis(self, analysis_result):
        logger.info("Starting analysis display")
        st.success("Document processed successfully!")
        
        # Get comprehensive ESG analysis
        comprehensive_query = """
        Analyze the document and provide a detailed extraction of all ESG metrics and data points.

        For each metric category:

        1. Environmental Metrics:
        - Extract exact values for emissions (Scope 1, 2, 3) with units
        - List all energy consumption figures with years
        - Include water and waste management metrics
        - Note all environmental compliance data
        - List any targets and current progress

        2. Social Metrics:
        - Provide workforce diversity percentages
        - List training hours and investments
        - Include safety incident rates and statistics
        - Detail community investment figures
        - Note human rights compliance metrics

        3. Governance Metrics:
        - List board composition percentages
        - Include ethics violation statistics
        - Detail risk management metrics
        - Provide executive compensation ratios
        - Note shareholder rights metrics

        For each data point:
        - Include the exact numerical value
        - Specify the unit of measurement
        - Note the time period/year
        - Include any relevant context or comparisons

        Present all information in a clear, structured format that emphasizes numerical data and time series where available.
        """
        
        with st.spinner("Generating comprehensive ESG analysis..."):
            try:
                logger.info("Sending comprehensive query to RAG")
                raw_response = analysis_result["qa_chain"].invoke(comprehensive_query)
                logger.info("Received RAG response")
                logger.debug(f"RAG Response: {raw_response}")
                
                # Format data for visualization
                logger.info("Formatting data for visualization")
                visualization_data = self.data_formatter.format_for_visualization(
                    analysis_result["qa_chain"],
                    raw_response
                )
                
                if visualization_data:
                    logger.info("Successfully created visualization data")
                    # Create dashboard...
                    
                else:
                    logger.error("Failed to structure analysis data")
                    st.error("Failed to structure the analysis data for visualization")
                    with st.expander("View Raw Analysis"):
                        st.write(raw_response)
                        
            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                st.error(f"Error during analysis: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")

    def _add_download_options(self, raw_response, visualization_data):
        st.sidebar.subheader("Download Options")
        
        # Ensure raw_response is a string
        if isinstance(raw_response, dict):
            raw_response = json.dumps(raw_response, indent=2)
        
        # Download raw analysis
        st.sidebar.download_button(
            label="Download Raw Analysis",
            data=raw_response,
            file_name="esg_analysis.txt",
            mime="text/plain"
        )
        
        # Download structured data
        st.sidebar.download_button(
            label="Download Structured Data",
            data=json.dumps(visualization_data, indent=2),
            file_name="esg_metrics.json",
            mime="application/json"
        )

        # Add timestamp to downloads
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.sidebar.markdown(f"_Generated at: {timestamp}_")

def main():
    try:
        app = ESGDashboardApp()
        app.run()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please refresh the page and try again.")
        # Log the error for debugging
        print(f"Error details: {str(e)}")

if __name__ == "__main__":
    main()