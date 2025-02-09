import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import Dict, List, Any, Optional, Union
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
import logging

# Set environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('esg_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

class ESGDashboardApp:
    def __init__(self):
        self.dashboard_creator = DashboardCreator()
        self.comprehensive_query = """
        Analyze the document and provide a detailed comprehensive analysis of all ESG metrics.
        Return the response as a clean JSON object without any markdown formatting or additional text.
        IMPORTANT: Return ONLY the raw JSON object without any markdown formatting (no ``` or ```json tags).

        Focus on:
        1. Environmental Metrics:
           - Emissions (Scope 1, 2, 3)
           - Energy consumption and efficiency
           - Water management
           - Waste management
           - Environmental compliance
           
        2. Social Metrics:
           - Workforce diversity and inclusion
           - Employee training and development
           - Health and safety metrics
           - Community engagement
           - Human rights
           
        3. Governance Metrics:
           - Board composition and diversity
           - Ethics and compliance
           - Risk management
           - Executive compensation
           - Shareholder rights
           
        For each metric:
        - Extract specific numerical values
        - Include time series data where available
        - Note targets and progress
        - Identify trends and patterns
        - Compare with previous years
        
        Format the response as a structured JSON with the following format:
        {
            "environmental": {
                "metrics": [
                    {
                        "id": "env_emissions",
                        "title": "Carbon Emissions",
                        "visualization": {
                            "type": "bar",
                            "data": {
                                "x": ["Scope 1", "Scope 2", "Scope 3"],
                                "y": [actual_values],
                                "units": "tCO2e"
                            },
                            "properties": {
                                "x_title": "Emission Scope",
                                "y_title": "Value",
                                "show_legend": true
                            }
                        },
                        "insights": "Key findings about this metric"
                    }
                ],
                "kpis": [
                    {
                        "title": "Total Emissions",
                        "value": "actual_value tCO2e",
                        "trend": "positive/negative/neutral",
                        "comparison": "vs previous year"
                    }
                ]
            },
            "social": {
                "metrics": [],
                "kpis": []
            },
            "governance": {
                "metrics": [],
                "kpis": []
            }
        }

        Rules:
        1. Extract ALL numerical values with their units
        2. Use appropriate visualization types:
           - line charts for time series data
           - bar charts for comparisons
           - pie charts for percentages
        3. Include ONLY metrics with actual numerical values
        4. Always include units where available
        5. Add meaningful insights for each metric
        6. Ensure chronological ordering for time series data
        7. Group related metrics together

        Return ONLY the JSON structure, no additional text or explanations.
        Ensure all numerical values are properly formatted and include their units
        """
    
    def run(self):
        self._setup_page()
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
        st.success("Document processed successfully!")
        
        with st.spinner("Generating comprehensive ESG analysis..."):
            try:
                logger.info("Sending comprehensive query to LLM")
                raw_response = analysis_result["qa_chain"].invoke({"query": self.comprehensive_query})
                logger.info("Received LLM response")
                logger.debug(f"Raw response: {raw_response}")

                # Debug: Show raw response
                with st.expander("Debug: Raw Analysis", expanded=False):
                    st.json(raw_response)

                # Extract and parse the response
                try:
                    # Get the result content
                    if isinstance(raw_response, dict):
                        result_content = raw_response.get("result", "")
                    else:
                        result_content = str(raw_response)

                    # Clean up the response - remove markdown backticks and 'json' identifier
                    cleaned_content = result_content.replace("```json", "").replace("```", "").strip()
                    logger.debug(f"Cleaned content: {cleaned_content}")

                    # Parse the JSON
                    visualization_data = json.loads(cleaned_content)
                    logger.info("Successfully parsed JSON response")

                    # Debug: Show parsed data
                    with st.expander("Debug: Structured Data", expanded=False):
                        st.json(visualization_data)

                    # Create dashboard
                    self._create_dashboard(visualization_data)

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}")
                    st.error("Failed to parse the analysis data")
                    
                    # Show the problematic content for debugging
                    with st.expander("Raw Response Content", expanded=True):
                        st.text("Raw Response:")
                        st.write(raw_response)
                        st.text("Attempted to parse:")
                        if isinstance(raw_response, dict):
                            st.write(raw_response.get("result", "No result found"))
                        else:
                            st.write(raw_response)

            except Exception as e:
                logger.error(f"Error during analysis: {str(e)}")
                st.error(f"Error during analysis: {str(e)}")
                st.write("Please try again or contact support if the issue persists.")

    def _create_dashboard(self, data: Dict):
        """Create the dashboard with visualizations"""
        
        # Display summary metrics
        st.subheader("📊 Key Metrics Summary")
        self._display_summary_metrics(data)
        
        # Create main dashboard tabs
        env_tab, soc_tab, gov_tab = st.tabs([
            "Environmental 🌍", 
            "Social 👥", 
            "Governance ⚖️"
        ])
        
        with env_tab:
            self._display_category_metrics(data.get("environmental", {}), "environmental")
        
        with soc_tab:
            self._display_category_metrics(data.get("social", {}), "social")
        
        with gov_tab:
            self._display_category_metrics(data.get("governance", {}), "governance")

    def _display_summary_metrics(self, data: Dict):
        """Display summary KPIs"""
        for category in ['environmental', 'social', 'governance']:
            if category in data and 'kpis' in data[category] and data[category]['kpis']:
                st.subheader(f"{category.title()} KPIs")
                cols = st.columns(min(3, len(data[category]['kpis'])))
                
                for idx, kpi in enumerate(data[category]['kpis']):
                    with cols[idx % 3]:
                        # Convert trend to proper delta_color value
                        delta_color = "normal"  # default value
                        if kpi.get("trend") == "positive":
                            delta_color = "normal"  # green when positive
                        elif kpi.get("trend") == "negative":
                            delta_color = "inverse"  # red when negative
                        # "neutral" will keep the default "normal" color
                            
                        st.metric(
                            label=kpi.get('title', ''),
                            value=kpi.get('value', ''),
                            delta=kpi.get('comparison', None),
                            delta_color=delta_color
                        )

    def _display_category_metrics(self, category_data: Dict, category: str):
        """Display metrics for a specific category"""
        if not category_data:
            st.info(f"No {category.title()} metrics available")
            return
        
        # Display metrics and their visualizations
        if 'metrics' in category_data and category_data['metrics']:
            for metric in category_data['metrics']:
                with st.expander(metric.get('title', 'Metric'), expanded=True):
                    # Create visualization
                    viz_config = metric.get('visualization', {})
                    if viz_config:
                        fig = self._create_visualization(viz_config, category)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    if 'insights' in metric:
                        st.info(metric['insights'])

    def _create_visualization(self, viz_config: Dict, category: str) -> Optional[go.Figure]:
        """Create a Plotly visualization based on the configuration"""
        try:
            chart_type = viz_config.get('type', 'bar')
            data = viz_config.get('data', {})
            props = viz_config.get('properties', {})
            
            fig = go.Figure()
            
            if chart_type == 'bar':
                fig.add_trace(go.Bar(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    name=props.get('name', ''),
                    marker_color=self._get_category_color(category)
                ))
                
            elif chart_type == 'line':
                fig.add_trace(go.Scatter(
                    x=data.get('x', []),
                    y=data.get('y', []),
                    mode='lines+markers',
                    name=props.get('name', ''),
                    line=dict(color=self._get_category_color(category))
                ))
                
            elif chart_type == 'pie':
                fig = go.Figure(data=[go.Pie(
                    labels=data.get('x', []),
                    values=data.get('y', []),
                    hole=0.3
                )])
            
            # Update layout
            fig.update_layout(
                title=props.get('title', ''),
                xaxis_title=props.get('x_title', ''),
                yaxis_title=props.get('y_title', ''),
                showlegend=props.get('show_legend', True),
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                template='plotly_white'
            )
            
            # Add units to axis labels if available
            if 'units' in data:
                if props.get('y_title'):
                    fig.update_layout(yaxis_title=f"{props['y_title']} ({data['units']})")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None

    def _get_category_color(self, category: str) -> str:
        """Get color for category"""
        colors = {
            'environmental': '#2ecc71',
            'social': '#3498db',
            'governance': '#f1c40f'
        }
        return colors.get(category, '#95a5a6')

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

    def _display_kpis(self, kpis: List[Dict[str, Any]]):
        cols = st.columns(len(kpis))
        for col, kpi in zip(cols, kpis):
            with col:
                # Convert trend to proper delta_color value
                delta_color = "normal"  # default value
                if kpi.get("trend") == "positive":
                    delta_color = "normal"  # green when positive
                elif kpi.get("trend") == "negative":
                    delta_color = "inverse"  # red when negative
                # "neutral" will keep the default "normal" color
                
                st.metric(
                    label=kpi["title"],
                    value=kpi["value"],
                    delta=kpi.get("comparison"),
                    delta_color=delta_color
                )

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
            elif chart_type == "pie":
                return self._create_pie_chart(data, props, metric_data["title"], category)
            elif chart_type == "scatter":
                return self._create_scatter_chart(data, props, metric_data["title"], category)
            
        except Exception as e:
            logger.error(f"Error creating chart for {metric_data.get('id', 'unknown')}: {str(e)}")
            return None

    def _create_line_chart(self, data, props, title, category):
        fig = go.Figure()

        if "series" in data:
            for idx, series_name in enumerate(data["series"]):
                fig.add_trace(go.Scatter(
                    x=data["x"],
                    y=data["y"][idx] if isinstance(data["y"][0], list) else data["y"],
                    name=series_name,
                    mode='lines+markers',
                    line=dict(color=self.color_schemes[category][idx % len(self.color_schemes[category])])
                ))
        else:
            fig.add_trace(go.Scatter(
                x=data["x"],
                y=data["y"],
                mode='lines+markers',
                line=dict(color=self.color_schemes[category][0])
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

    def _create_pie_chart(self, data, props, title, category):
        fig = go.Figure(data=[go.Pie(
            labels=data["x"],
            values=data["y"],
            hole=0.3,
            marker=dict(colors=self.color_schemes[category])
        )])

        self._update_layout(fig, title, props)
        return fig

    def _create_scatter_chart(self, data, props, title, category):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode='markers',
            marker=dict(
                color=self.color_schemes[category][0],
                size=10
            )
        ))

        self._update_layout(fig, title, props, data.get("units"))
        return fig

    def _update_layout(self, fig, title, props, units=None):
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
                    logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text.strip():
                logger.error("No text could be extracted from the PDF")
                return None
                
            logger.info(f"Successfully extracted text from PDF: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return None

    def split_text(self, text: str) -> List[Dict[str, str]]:
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return [
                {"content": chunk, "source": f"chunk_{i}"} 
                for i, chunk in enumerate(chunks)
            ]
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            return []

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

    def create_store(self, texts: List[Dict[str, str]]) -> Optional[FAISS]:
        try:
            logger.info(f"Creating vector store with {len(texts)} texts")
            vectorstore = FAISS.from_texts(
                texts=[t["content"] for t in texts],
                embedding=self.embeddings,
                metadatas=texts
            )
            
            os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
            vectorstore.save_local(self.store_path)
            logger.info(f"Vector store saved to {self.store_path}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None

    def load_store(self) -> Optional[FAISS]:
        try:
            if os.path.exists(self.store_path):
                logger.info(f"Loading vector store from {self.store_path}")
                return FAISS.load_local(
                    self.store_path,
                    self.embeddings
                )
            logger.warning(f"Vector store not found at {self.store_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
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
        logger.info("ESGAnalyzer initialized")

    def create_qa_chain(self, vectorstore: FAISS) -> RetrievalQA:
        try:
            logger.info("Creating QA chain")
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                ),
                return_source_documents=False
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise

    def process_document(self, file_content: bytes, is_private: bool = False) -> Optional[Dict[str, Any]]:
        try:
            logger.info("Starting document processing")
            
            # Process PDF
            text = self.doc_processor.process_pdf(file_content)
            if not text:
                logger.error("Failed to process PDF")
                return None
            
            # Create vector store
            store_name = f"esg_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chunks = self.doc_processor.split_text(text)
            vector_manager = VectorStoreManager(store_name, is_private)
            vectorstore = vector_manager.create_store(chunks)
            
            if not vectorstore:
                logger.error("Failed to create vector store")
                return None
            
            # Create QA chain
            qa_chain = self.create_qa_chain(vectorstore)
            
            logger.info("Document processing completed successfully")
            return {
                "qa_chain": qa_chain,
                "text": text,
                "vectorstore": vectorstore
            }
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return None

def main():
    try:
        logger.info("Starting ESG Analysis Dashboard")
        app = ESGDashboardApp()
        app.run()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
        st.write("Please refresh the page and try again.")

if __name__ == "__main__":
    main()