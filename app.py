import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import time
from datetime import date
import pickle
import itertools
import plotly.express as px
from plot_setup import finastra_theme
from merge_data import Data
import sys
import metadata_parser
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client with proper error handling
def initialize_groq_client():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.warning("Groq API key not found. Some insights features will be disabled.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {str(e)}")
        return None

groq_client = initialize_groq_client()

def get_graph_insight(chart_data, chart_type):
    if not groq_client:
        return "Insights not available - Groq API key not configured."
        
    prompt = f"""
    Analyze the following data for a {chart_type} and provide a brief, insightful observation:
    {chart_data}
    
    Instructions:
    - Provide 1-2 sentences of key insights
    - Focus on trends, patterns, or notable observations
    - Be specific and data-driven
    - Keep it concise and business-relevant
    """
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert ESG and financial analyst with the following capabilities:
                    Core Expertise:
                    - Advanced interpretation of ESG metrics and sustainability trends
                    - Deep understanding of financial markets and corporate performance
                    - Specialized knowledge in environmental impact, social responsibility, and corporate governance
                    - Expert analysis of sentiment patterns and market dynamics"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-70b-8192",
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Unable to generate insight: {str(e)}"

# Set page config
st.set_page_config(
    page_title="Sustain | ESG Analytics",
    page_icon="🌱",
    layout='wide',
    initial_sidebar_state="expanded"
)

####### CACHED FUNCTIONS ######
@st.cache_data(show_spinner=False)
def filter_company_data(df_company, esg_categories, start, end):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    comps = []
    for i in esg_categories:
        X = df_company[df_company[i] == True]
        comps.append(X)
    df_company = pd.concat(comps)
    df_company = df_company[df_company.DATE.between(start, end)]
    return df_company

@st.cache_data(show_spinner=False)
def load_data(start_data, end_data):
    data = Data().read(start_data, end_data)
    if 'data' in data and 'DATE' in data['data'].columns:
        data['data']['DATE'] = pd.to_datetime(data['data']['DATE'])
    companies = data["data"].Organization.sort_values().unique().tolist()
    companies.insert(0,"Select a Company")
    return data, companies

@st.cache_data(show_spinner=False)
def filter_publisher(df_company, publisher):
    if publisher != 'all':
        df_company = df_company[df_company['SourceCommonName'] == publisher]
    return df_company
def get_melted_frame(data_dict, frame_names, keepcol=None, dropcol=None):
    try:
        if keepcol:
            reduced = {k: df[keepcol].rename(k) for k, df in data_dict.items() 
                       if k in frame_names and keepcol in df.columns}
        else:
            reduced = {k: df.drop(columns=dropcol).mean(axis=1).rename(k) 
                       for k, df in data_dict.items() if k in frame_names and dropcol in df.columns}
        
        if not reduced:
            raise ValueError(f"Column '{keepcol if keepcol else dropcol}' not found in the dataset.")
        
        df = (pd.concat(list(reduced.values()), axis=1)
              .reset_index()
              .melt("date")
              .sort_values("date")
              .ffill())
        df.columns = ["DATE", "ESG", "Score"]
        return df.reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error in get_melted_frame: {str(e)}")
        return pd.DataFrame()

def filter_on_date(df, start, end, date_col="DATE"):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]
    return df

@st.cache_data(show_spinner=False)
def get_clickable_name(url):
    try:
        T = metadata_parser.MetadataParser(url=url, search_head_only=True)
        title = T.metadata["og"]["title"].replace("|", " - ")
        return f"[{title}]({url})"
    except:
        return f"[{url}]({url})"

def main(start_data, end_data):
    # Set up theme
    alt.themes.register("finastra", finastra_theme)
    alt.themes.enable("finastra")
    violet, fuchsia = ["#694ED6", "#C137A2"]

    # Header
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #694ED6;'>Sustain | ESG Analytics Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner(text="Fetching Data..."):
        try:
            data, companies = load_data(start_data, end_data)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return

    df_conn = data["conn"]
    df_data = data["data"]
    embeddings = data["embed"]

    # Sidebar Configuration
    st.sidebar.markdown("""
        <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 8px;'>
            <h3 style='color: #694ED6;'>Dashboard Controls</h3>
        </div>
    """, unsafe_allow_html=True)
    
    date_place = st.sidebar.empty()
    esg_categories = st.sidebar.multiselect(
        "ESG Categories",
        ["E", "S", "G"],
        ["E", "S", "G"],
        help="Select Environmental, Social, and/or Governance categories"
    )
    
    pub = st.sidebar.empty()
    num_neighbors = st.sidebar.slider(
        "Connection Depth",
        1, 20, 8,
        help="Adjust the number of connected companies to display"
    )

    # Main company selector
    company = st.selectbox(
        "Select Company for Analysis",
        companies,
        help="Choose a company to view its ESG analytics"
    )
    if company and company != "Select a Company":
        # Data preparation
        df_company = df_data[df_data.Organization == company]
        diff_col = f"{company.replace(' ', '_')}_diff"
        esg_keys = ["E_score", "S_score", "G_score"]
        esg_df = get_melted_frame(data, esg_keys, keepcol=diff_col)
        ind_esg_df = get_melted_frame(data, esg_keys, dropcol="industry_tone")
        tone_df = get_melted_frame(data, ["overall_score"], keepcol=diff_col)
        ind_tone_df = get_melted_frame(data, ["overall_score"], dropcol="industry_tone")

        # Date range setup
        if isinstance(df_company.DATE.min(), pd.Timestamp):
            min_date = df_company.DATE.min().date()
            max_date = df_company.DATE.max().date()
        else:
            min_date = df_company.DATE.min()
            max_date = df_company.DATE.max()

        selected_dates = date_place.date_input(
            "Analysis Period",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )
        start, end = selected_dates

        # Apply filters
        df_company = filter_company_data(df_company, esg_categories, start, end)
        esg_df = filter_on_date(esg_df, start, end)
        ind_esg_df = filter_on_date(ind_esg_df, start, end)
        tone_df = filter_on_date(tone_df, start, end)
        ind_tone_df = filter_on_date(ind_tone_df, start, end)

        # Publisher filter
        publishers = df_company.SourceCommonName.sort_values().unique().tolist()
        publishers.insert(0, "all")
        publisher = pub.selectbox("News Source Filter", publishers)
        df_company = filter_publisher(df_company, publisher)

        # Key Metrics Dashboard
        st.markdown("---")
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            st.metric(
                "Total Articles",
                f"{len(df_company):,}",
                help="Total number of articles analyzed"
            )
        
        with metrics_col2:
            avg_tone = df_company['Tone'].mean()
            st.metric(
                "Average Tone",
                f"{avg_tone:.2f}",
                help="Average sentiment tone (-10 to +10)"
            )
        
        with metrics_col3:
            avg_esg = esg_df['Score'].mean()
            st.metric(
                "ESG Score",
                f"{avg_esg:.2f}",
                help="Overall ESG performance score"
            )
        
        with metrics_col4:
            coverage_days = (end - start).days
            st.metric(
                "Analysis Period",
                f"{coverage_days} days",
                help="Time period covered in analysis"
            )

        # Article Details Section
        with st.expander("📰 Article Analysis Details", expanded=True):
            st.markdown(f"### News Coverage Analysis for {company.title()}")
            
            # Article metrics in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                display_cols = ["DATE", "SourceCommonName", "Tone", "Polarity",
                              "NegativeTone", "PositiveTone"]
                st.dataframe(
                    df_company[display_cols],
                    use_container_width=True,
                    height=200
                )
            
            with col2:
                st.markdown("#### Key Articles")
                link_df = df_company[["DATE", "DocumentIdentifier"]].head(3).copy()
                link_df["ARTICLE"] = link_df.DocumentIdentifier.apply(get_clickable_name)
                link_df = link_df[["DATE", "ARTICLE"]].to_markdown(index=False)
                st.markdown(link_df)

        # Main Dashboard Layout
        st.markdown("---")
        st.markdown("### ESG Performance Analysis")
        
        # Three-column layout for main visualizations
        viz_col1, viz_col2, viz_col3 = st.columns([1, 1, 1])
        # ESG Radar Chart
        with viz_col1:
            st.markdown("#### ESG Score Breakdown")
            avg_esg = data["ESG"]
            avg_esg.rename(columns={"Unnamed: 0": "Type"}, inplace=True)
            avg_esg.replace({
                "T": "Overall",
                "E": "Environment",
                "S": "Social",
                "G": "Governance"
            }, inplace=True)

            numeric_cols = avg_esg.columns.difference(['Type'])
            avg_esg[numeric_cols] = avg_esg[numeric_cols].apply(pd.to_numeric, errors='coerce')
            avg_esg["Industry Average"] = avg_esg[numeric_cols].mean(axis=1)

            radar_df = avg_esg[["Type", company, "Industry Average"]].melt(
                "Type",
                value_name="score",
                var_name="entity"
            )

            radar = px.line_polar(
                radar_df,
                r="score",
                theta="Type",
                color="entity",
                line_close=True,
                hover_name="Type",
                hover_data={
                    "Type": True,
                    "entity": True,
                    "score": ":.2f"
                },
                color_discrete_map={
                    "Industry Average": fuchsia,
                    company: violet
                }
            )
            
            radar.update_layout(
                template=None,
                polar={
                    "radialaxis": {"showticklabels": False, "ticks": ""},
                    "angularaxis": {"showticklabels": True, "ticks": ""}
                },
                legend={
                    "title": None,
                    "yanchor": "middle",
                    "orientation": "h"
                },
                margin={"l": 10, "r": 10, "t": 30, "b": 10},
                height=400
            )
            
            st.plotly_chart(radar, use_container_width=True)
    

        # Tone Distribution
        with viz_col2:
            st.markdown("#### Sentiment Distribution")
            dist_chart = alt.Chart(
                df_company,
                title="Document Tone Distribution"
            ).transform_density(
                density='Tone',
                as_=["Tone", "density"]
            ).mark_area(
                opacity=0.6,
                color=violet
            ).encode(
                x=alt.X('Tone:Q', scale=alt.Scale(domain=(-10, 10))),
                y='density:Q',
                tooltip=[
                    alt.Tooltip("Tone", format=".3f"),
                    alt.Tooltip("density:Q", format=".4f")
                ]
            ).properties(
                height=350
            ).interactive()
            
            st.altair_chart(dist_chart, use_container_width=True)
            

        # Metric Over Time
                # Metric Over Time
        with viz_col3:
            st.markdown("#### Temporal Analysis")
            metric_options = [
                "Tone",
                "NegativeTone",
                "PositiveTone",
                "Polarity",
                "Overall Score",
                "ESG Scores"
            ]
            line_metric = st.radio("Select Metric", options=metric_options)
            
            if line_metric == "ESG Scores":
                esg_df["WHO"] = company.title()
                ind_esg_df["WHO"] = "Industry Average"
                plot_df = pd.concat([esg_df, ind_esg_df]).reset_index(drop=True)
                plot_df.replace({
                    "E_score": "Environment",
                    "S_score": "Social",
                    "G_score": "Governance"
                }, inplace=True)

                metric_chart = alt.Chart(
                    plot_df,
                    title="ESG Trends"
                ).mark_line().encode(
                    x=alt.X("yearmonthdate(DATE):O", title="Date"),
                    y=alt.Y("Score:Q"),
                    color=alt.Color(
                        "ESG",
                        sort=None,
                        legend=alt.Legend(title=None, orient="top")
                    ),
                    strokeDash=alt.StrokeDash(
                        "WHO",
                        sort=None,
                        legend=alt.Legend(
                            title=None,
                            symbolType="stroke",
                            orient="top"
                        )
                    ),
                    tooltip=[
                        "DATE",
                        "ESG",
                        alt.Tooltip("Score", format=".5f")
                    ]
                )
            else:
                if line_metric == "Overall Score":
                    line_metric = "Score"
                    tone_df["WHO"] = company.title()
                    ind_tone_df["WHO"] = "Industry Average"
                    plot_df = pd.concat([tone_df, ind_tone_df]).reset_index(drop=True)
                else:
                    df1 = df_company.groupby("DATE")[line_metric].mean().reset_index()
                    df2 = filter_on_date(
                        df_data.groupby("DATE")[line_metric].mean().reset_index(),
                        start,
                        end
                    )
                    df1["WHO"] = company.title()
                    df2["WHO"] = "Industry Average"
                    plot_df = pd.concat([df1, df2]).reset_index(drop=True)
                
                metric_chart = alt.Chart(
                    plot_df,
                    title=f"{line_metric} Trends"
                ).mark_line().encode(
                    x=alt.X("yearmonthdate(DATE):O", title="Date"),
                    y=alt.Y(f"{line_metric}:Q", scale=alt.Scale(type="linear")),
                    color=alt.Color(
                        "WHO",
                        legend=alt.Legend(title=None, orient="top")
                    ),
                    tooltip=[
                        "DATE",
                        alt.Tooltip(line_metric, format=".3f")
                    ]
                )

            metric_chart = metric_chart.properties(
                height=350
            ).interactive()
            
            st.altair_chart(metric_chart, use_container_width=True)


        # Company Connections Section
        st.markdown("---")
        st.markdown("### 🔗 Company Network Analysis")
        
        company_df = df_conn[df_conn.company == company]
        similar_org_cols = [col for col in df_conn.columns if 'similar_org' in col]
        similarity_cols = [col for col in df_conn.columns if 'similarity' in col]
        
        similar_org_cols = similar_org_cols[:num_neighbors]
        similarity_cols = similarity_cols[:num_neighbors]

        if not company_df.empty and all(col in company_df.columns for col in similar_org_cols + similarity_cols):
            conn_col1, conn_col2 = st.columns([2, 1])
            
            with conn_col1:
                neighbors = company_df[similar_org_cols].iloc[0]
                neighbor_confidences = company_df[similarity_cols].iloc[0]
                
                color_f = lambda f: f"Company: {company.title()}" if f == company else (
                    "Connected Company" if f in neighbors.values else "Other Company")
                embeddings["colorCode"] = embeddings.company.apply(color_f)
                
                point_colors = {
                    f"Company: {company.title()}": violet,
                    "Connected Company": fuchsia,
                    "Other Company": "#e0e0e0"
                }
                
                fig_3d = px.scatter_3d(
                    embeddings,
                    x="1",
                    y="2",
                    z="3",
                    color='colorCode',
                    color_discrete_map=point_colors,
                    opacity=0.7,
                    hover_name="company"
                )
                
                fig_3d.update_layout(
                    scene={
                        "xaxis": {"visible": False},
                        "yaxis": {"visible": False},
                        "zaxis": {"visible": False}
                    },
                    margin={"l": 0, "r": 0, "t": 0, "b": 0},
                    legend={
                        "title": None,
                        "orientation": "h",
                        "yanchor": "bottom",
                        "y": 1.02
                    },
                    height=500
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                embedding_insight = get_graph_insight(
                    {
                        "company": company,
                        "neighbors": neighbors.to_dict(),
                        "connections": len(neighbors)
                    },
                    "Company Network Analysis"
                )
                st.markdown(f"**Network Insights:** {embedding_insight}")
            with conn_col2:
                st.markdown("#### Connection Strength")
                neighbor_conf = pd.DataFrame({
                    "Neighbor": neighbors.values,
                    "Confidence": neighbor_confidences.values
                })
                
                conf_plot = alt.Chart(
                    neighbor_conf,
                    title="Connected Companies"
                ).mark_bar().encode(
                    x=alt.X("Confidence:Q", title="Similarity Score"),
                    y=alt.Y("Neighbor:N", sort="-x", title=None),
                    tooltip=["Neighbor", alt.Tooltip("Confidence", format=".3f")],
                    color=alt.Color(
                        "Confidence:Q",
                        scale=alt.Scale(scheme='purples'),
                        legend=None
                    )
                ).properties(
                    height=25 * num_neighbors + 50
                ).configure_axis(
                    grid=False
                )
                
                st.altair_chart(conf_plot, use_container_width=True)
                


        else:
            st.warning("No connection data available for this company.")

        

        # Overall Analysis Summary
        st.markdown("---")
        st.markdown("### Comprehensive Performance Overview")
        
        st.markdown(f"""
            #### Key Metrics for {company.title()}
            
            **News Sentiment Analysis**
            - Total Articles: {len(df_company):,}
            - Coverage Period: {start} to {end}
            - Average Sentiment: {df_company['Tone'].mean():.2f}
            - Sentiment Distribution:
              • Positive: {(df_company['Tone'] > 0).mean():.1%}
              • Negative: {(df_company['Tone'] < 0).mean():.1%}
              • Neutral: {(df_company['Tone'] == 0).mean():.1%}
            - News Sources: {len(df_company['SourceCommonName'].unique())}

            **Network Analysis**
            - Connected Companies: {num_neighbors}
            - Strongest Connection: {neighbor_conf['Neighbor'].iloc[0]} 
            - Connection Strength: {neighbor_conf['Confidence'].iloc[0]:.2f}
            - Total Network Interactions: {len(df_company):,}
        """)


if __name__ == "__main__":
    try:
        args = sys.argv
        if len(args) != 3:
            start_data = "jan1"
            end_data = "jan20"
        else:
            start_data = args[1]
            end_data = args[2]
        
        main(start_data, end_data)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        alt.themes.enable("default")
