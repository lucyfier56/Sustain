from typing import List, Dict, Optional, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import io
import os
import re
from config import settings

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
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
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'}
        )
        self.store_path = os.path.join(
            settings.VECTOR_DB_PATH,
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

class DashboardEngine:
    def __init__(self):
        self.predefined_queries = settings.PREDEFINED_QUERIES

    def generate_visualization_data(self, qa_chain, metric_type: str) -> Dict[str, Any]:
        try:
            query = self.predefined_queries.get(f"{metric_type}_metrics")
            if not query:
                return {}
                
            response = qa_chain.run(query)
            
            # Parse metrics from response
            metrics = self._extract_metrics(response)
            
            return {
                "raw_data": response,
                "parsed_metrics": metrics,
                "metric_type": metric_type,
                "visualization_type": self._get_recommended_viz_type(metric_type),
                "chart_data": self._prepare_chart_data(metrics)
            }
        except Exception as e:
            print(f"Error generating visualization data: {str(e)}")
            return {}

    def _extract_metrics(self, text: str) -> Dict[str, Any]:
        metrics = {
            "numerical_values": re.findall(r'\d+(?:\.\d+)?', text),
            "percentages": re.findall(r'\d+(?:\.\d+)?%', text),
            "monetary": re.findall(r'[\$€£]\s*\d+(?:\.\d+)?(?:\s*[MBK])?', text)
        }
        return metrics

    def _get_recommended_viz_type(self, metric_type: str) -> str:
        viz_types = settings.CHART_TYPES.get(metric_type, ["bar"])
        return viz_types[0]

    def _prepare_chart_data(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        # Convert extracted metrics into chart-ready format
        return {
            "labels": [f"Metric {i+1}" for i in range(len(metrics.get("numerical_values", [])))],
            "values": metrics.get("numerical_values", []),
            "percentages": metrics.get("percentages", []),
            "monetary": metrics.get("monetary", [])
        }

class RAGCore:
    def __init__(self, groq_api_key: str):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name=settings.LLM_MODEL,
            max_tokens=4096,
            api_key=groq_api_key
        )
        self.doc_processor = DocumentProcessor()

    def create_qa_chain(self, vectorstore: FAISS) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": settings.TOP_K_RESULTS}
            ),
            return_source_documents=False
        )

    def analyze_document(self, text: str, store_name: str, is_private: bool = False) -> Optional[Dict[str, Any]]:
        try:
            chunks = self.doc_processor.split_text(text)
            vector_manager = VectorStoreManager(store_name, is_private)
            vectorstore = vector_manager.create_store(chunks)
            return {
                "qa_chain": self.create_qa_chain(vectorstore),
                "vectorstore": vectorstore,
                "text": text
            }
        except Exception as e:
            print(f"Error in document analysis: {str(e)}")
            return None

    def query(self, qa_chain: RetrievalQA, prompt: str) -> Optional[str]:
        try:
            return qa_chain.run(prompt)
        except Exception as e:
            print(f"Error in query: {str(e)}")
            return None