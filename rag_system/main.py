import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from analyzer import ESGAnalyzer
from config import settings
import os
import re

def create_visualization(data: dict, metric_type: str):
    if not data or not data.get("chart_data"):
        return None
        
    chart_data = data["chart_data"]
    viz_type = data["visualization_type"]
    
    df = pd.DataFrame({
        "Metric": chart_data["labels"],
        "Value": [float(v) for v in chart_data["values"] if v.replace('.','').isdigit()]
    })
    
    if viz_type == "line":
        fig = px.line(df, x="Metric", y="Value", title=f"{metric_type.title()} Metrics")
    elif viz_type == "bar":
        fig = px.bar(df, x="Metric", y="Value", title=f"{metric_type.title()} Metrics")
    elif viz_type == "scatter":
        fig = px.scatter(df, x="Metric", y="Value", title=f"{metric_type.title()} Metrics")
    else:
        fig = px.bar(df, x="Metric", y="Value", title=f"{metric_type.title()} Metrics")
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def display_validation_results(validation_results):
    st.subheader("Document Validation Results")
    
    # Display scores
    scores = validation_results["scores"]
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Overall Score", f"{scores['overall']}%")
    with cols[1]:
        st.metric("Completeness", f"{scores['completeness']}%")
    with cols[2]:
        st.metric("Data Quality", f"{scores['data_quality']}%")
    with cols[3]:
        st.metric("Consistency", f"{scores['consistency']}%")
    
    # Detailed results in expandable sections
    with st.expander("Detailed Validation Results"):
        # Completeness Check
        st.write("**Section Completeness:**")
        for section, present in validation_results["completeness"].items():
            st.write(f"- {section.title()}: {'✅' if present else '❌'}")
        
        # Data Quality
        st.write("\n**Data Quality Metrics:**")
        for metric, details in validation_results["data_quality"].items():
            st.write(f"- {metric.title()}:")
            st.write(f"  - Count: {details['count']}")
            if details['examples']:
                examples = [str(ex) if isinstance(ex, tuple) else ex for ex in details['examples']]
                st.write(f"  - Examples: {', '.join(examples)}")
        
        # Consistency
        st.write("\n**Consistency Checks:**")
        for check, passed in validation_results["consistency"].items():
            st.write(f"- {check.replace('_', ' ').title()}: {'✅' if passed else '❌'}")

def create_esg_dashboard(qa_chain):
    comprehensive_query = """
    Provide a detailed comprehensive analysis on Environmental Social and Governance aspects.

    Environmental Analysis:
    1. Carbon emissions (Scope 1, 2, 3) and targets
    2. Energy consumption and renewable energy usage
    3. Water management and efficiency
    4. Waste management and recycling rates
    5. Environmental compliance and incidents

    Social Analysis:
    1. Workforce diversity statistics and targets
    2. Employee training hours and development programs
    3. Health and safety incidents and rates
    4. Community engagement and social investment
    5. Human rights and labor practices

    Governance Analysis:
    1. Board composition and diversity
    2. Ethics and compliance programs
    3. Risk management framework
    4. Stakeholder engagement practices
    5. Executive compensation and ESG links

    Format the response with clear separation between sections and include all available metrics, 
    year-over-year comparisons, and specific targets.
    """

    with st.spinner("Generating comprehensive ESG analysis..."):
        response = qa_chain.run(comprehensive_query)
        
        if response:
            # Create dashboard sections
            st.subheader("📊 ESG Performance Dashboard")
            
            # Create tabs for different sections
            env_tab, soc_tab, gov_tab = st.tabs(["Environmental", "Social", "Governance"])
            
            # Parse and display metrics
            metrics = parse_esg_metrics(response)
            
            with env_tab:
                st.subheader("🌍 Environmental Metrics")
                display_environmental_metrics(metrics.get('environmental', {}))
                st.markdown("### Detailed Environmental Analysis")
                st.markdown(extract_section(response, "Environmental"))
                
            with soc_tab:
                st.subheader("👥 Social Metrics")
                display_social_metrics(metrics.get('social', {}))
                st.markdown("### Detailed Social Analysis")
                st.markdown(extract_section(response, "Social"))
                
            with gov_tab:
                st.subheader("⚖️ Governance Metrics")
                display_governance_metrics(metrics.get('governance', {}))
                st.markdown("### Detailed Governance Analysis")
                st.markdown(extract_section(response, "Governance"))

def extract_section(text: str, section_name: str) -> str:
    sections = text.split('\n\n')
    for i, section in enumerate(sections):
        if section_name in section:
            # Try to get the section and the next few paragraphs
            return '\n\n'.join(sections[i:i+3])
    return "No detailed analysis available."

def parse_esg_metrics(response: str) -> dict:
    metrics = {
        'environmental': {},
        'social': {},
        'governance': {}
    }
    
    sections = response.split('\n\n')
    current_section = None
    
    for section in sections:
        if 'Environmental' in section:
            current_section = 'environmental'
        elif 'Social' in section:
            current_section = 'social'
        elif 'Governance' in section:
            current_section = 'governance'
        
        if current_section:
            numbers = re.findall(r'(\d+(?:\.\d+)?(?:\s*%|\s*tCO2e|\s*MWh|\s*m³)?)\s*(?:[-–]\s*([^,\n]+))?', section)
            if numbers:
                metrics[current_section][f'metric_{len(metrics[current_section])}'] = numbers
    
    return metrics

def display_environmental_metrics(metrics: dict):
    if not metrics:
        st.info("No environmental metrics found")
        return
        
    fig = go.Figure()
    
    for metric_name, values in metrics.items():
        numbers = [float(re.findall(r'\d+(?:\.\d+)?', val[0])[0]) for val in values if re.findall(r'\d+(?:\.\d+)?', val[0])]
        labels = [val[1] if len(val) > 1 else f"Metric {i}" for i, val in enumerate(values)]
        
        if numbers and labels:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=labels[:len(numbers)],
                y=numbers,
                text=numbers,
                textposition='auto',
            ))
    
    fig.update_layout(
        title="Environmental Performance Metrics",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_social_metrics(metrics: dict):
    if not metrics:
        st.info("No social metrics found")
        return
        
    fig = go.Figure()
    
    for metric_name, values in metrics.items():
        numbers = [float(re.findall(r'\d+(?:\.\d+)?', val[0])[0]) for val in values if re.findall(r'\d+(?:\.\d+)?', val[0])]
        labels = [val[1] if len(val) > 1 else f"Metric {i}" for i, val in enumerate(values)]
        
        if numbers and labels:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=labels[:len(numbers)],
                y=numbers,
                text=numbers,
                textposition='auto',
            ))
    
    fig.update_layout(
        title="Social Performance Metrics",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_governance_metrics(metrics: dict):
    if not metrics:
        st.info("No governance metrics found")
        return
        
    fig = go.Figure()
    
    for metric_name, values in metrics.items():
        numbers = [float(re.findall(r'\d+(?:\.\d+)?', val[0])[0]) for val in values if re.findall(r'\d+(?:\.\d+)?', val[0])]
        labels = [val[1] if len(val) > 1 else f"Metric {i}" for i, val in enumerate(values)]
        
        if numbers and labels:
            fig.add_trace(go.Bar(
                name=metric_name,
                x=labels[:len(numbers)],
                y=numbers,
                text=numbers,
                textposition='auto',
            ))
    
    fig.update_layout(
        title="Governance Performance Metrics",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="ESG Report Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 ESG Report Analysis Dashboard")
    st.markdown("""
    Upload a sustainability report (PDF) to analyze its ESG metrics, initiatives, and commitments.
    """)

    with st.sidebar:
        st.header("Configuration")
        groq_api_key = st.text_input("Enter your Groq API key:", type="password")
        st.markdown("[Get Groq API key](https://console.groq.com/)")
        
        is_private = st.checkbox("Private Document", 
                               help="Enable for private document analysis")
        
        if st.button("Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    analyzer = ESGAnalyzer(groq_api_key)
    uploaded_file = st.file_uploader("Upload Sustainability Report (PDF)", type="pdf")

    if uploaded_file:
        st.info(f"Processing: {uploaded_file.name}")
        
        with st.spinner("Processing report..."):
            analysis_result = analyzer.process_document(uploaded_file.read(), is_private)
            
            if analysis_result:
                st.success("PDF processed successfully!")
                
                # Display validation results
                display_validation_results(analysis_result["validation"])
                
                # Create comprehensive ESG dashboard
                create_esg_dashboard(analysis_result["qa_chain"])
                
                # Detailed Analysis Tabs
                tabs = st.tabs([f"{details['icon']} {name}" 
                              for name, details in settings.ANALYSIS_ASPECTS.items()])

                for tab, aspect_name in zip(tabs, settings.ANALYSIS_ASPECTS.keys()):
                    with tab:
                        st.subheader(f"{settings.ANALYSIS_ASPECTS[aspect_name]['icon']} {aspect_name}")
                        with st.spinner(f"Analyzing {aspect_name.lower()}..."):
                            response = analyzer.analyze_aspect(
                                analysis_result["qa_chain"], 
                                aspect_name
                            )
                            if response:
                                st.markdown(response)

                # Summary and Recommendations
                st.subheader("📋 Summary and Recommendations")
                with st.spinner("Generating summary and recommendations..."):
                    summary_response = analyzer.generate_summary(analysis_result["qa_chain"])
                    if summary_response:
                        st.markdown(summary_response)

                # Interactive Q&A
                st.subheader("❓ Ask Specific Questions")
                user_question = st.text_input(
                    "Enter your question about the sustainability report:",
                    placeholder="e.g., What are the company's carbon emission targets?"
                )
                
                if user_question:
                    with st.spinner("Finding answer..."):
                        response = analyzer.custom_query(
                            analysis_result["qa_chain"], 
                            user_question
                        )
                        if response:
                            st.markdown(response)

if __name__ == "__main__":
    main()