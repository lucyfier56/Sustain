#IMPLEMENTATION - SANJAY
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import io

class ESGAnalyzer:
    def __init__(self, groq_api_key: str):
        self.setup_components(groq_api_key)
        self.setup_prompts()

    def setup_components(self, groq_api_key: str):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.3-70b-versatile",
            max_tokens=4096,
            api_key=groq_api_key
        )
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )

    def setup_prompts(self):
        self.analysis_aspects = {
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

    def process_pdf(self, uploaded_file):
        try:
            pdf_bytes = uploaded_file.read()
            pdf_stream = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\nPage {page_num + 1}:\n{page_text}"
                except Exception as e:
                    st.warning(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue
            
            if not text.strip():
                st.error("No text could be extracted from the PDF.")
                return None
                
            return text
            
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return None

    def create_qa_chain(self, text: str):
        try:
            chunks = self.text_splitter.split_text(text)
            
            if not chunks:
                st.error("No text chunks were created. The document might be empty or unreadable.")
                return None
            
            # Create FAISS vector store with metadata
            texts_with_metadata = [
                {"content": chunk, "source": f"chunk_{i}"} 
                for i, chunk in enumerate(chunks)
            ]
            
            vectorstore = FAISS.from_texts(
                texts=[t["content"] for t in texts_with_metadata],
                embedding=self.embeddings,
                metadatas=texts_with_metadata
            )
            
            # Create retrieval chain without sources
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                ),
                return_source_documents=False
            )
            
            return qa_chain
            
        except Exception as e:
            st.error(f"Error creating QA chain: {str(e)}")
            return None

    def analyze_aspect(self, qa_chain, prompt):
        try:
            response = qa_chain.run(prompt)
            return response
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            return None

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

    if not groq_api_key:
        st.warning("Please enter your Groq API key in the sidebar to continue.")
        return

    analyzer = ESGAnalyzer(groq_api_key)
    uploaded_file = st.file_uploader("Upload Sustainability Report (PDF)", type="pdf")

    if uploaded_file:
        st.info(f"Processing: {uploaded_file.name}")
        
        with st.spinner("Processing report..."):
            text = analyzer.process_pdf(uploaded_file)
            
            if text:
                st.success("PDF processed successfully!")
                
                with st.spinner("Creating analysis chain..."):
                    qa_chain = analyzer.create_qa_chain(text)
                
                if qa_chain:
                    tabs = st.tabs([f"{details['icon']} {name}" 
                                  for name, details in analyzer.analysis_aspects.items()])

                    for tab, (aspect_name, aspect_details) in zip(tabs, analyzer.analysis_aspects.items()):
                        with tab:
                            st.subheader(f"{aspect_details['icon']} {aspect_name}")
                            with st.spinner(f"Analyzing {aspect_name.lower()}..."):
                                response = analyzer.analyze_aspect(qa_chain, aspect_details['prompt'])
                                if response:
                                    st.markdown(response)

                    st.subheader("📋 Summary and Recommendations")
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
                    
                    with st.spinner("Generating summary and recommendations..."):
                        summary_response = analyzer.analyze_aspect(qa_chain, summary_prompt)
                        if summary_response:
                            st.markdown(summary_response)

                    st.subheader("❓ Ask Specific Questions")
                    user_question = st.text_input(
                        "Enter your question about the sustainability report:",
                        placeholder="e.g., What are the company's carbon emission targets?"
                    )
                    
                    if user_question:
                        with st.spinner("Finding answer..."):
                            response = analyzer.analyze_aspect(qa_chain, user_question)
                            if response:
                                st.markdown(response)

if __name__ == "__main__":
    main()

