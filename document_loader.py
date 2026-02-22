"""
Document loader and preprocessing for RAG pipeline
"""
import uuid
import hashlib
import re
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentLoader:
    """Load and prepare documents for retrieval"""
    
    @staticmethod
    def load_pdf(path: str) -> List[Document]:
        """
        Load documents from PDF file.
        
        Args:
            path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(path)
        docs = loader.load()
        return docs
    
    @staticmethod
    def load_documents(file_path: str = None, sample_docs: List[str] = None) -> List[Document]:
        """
        Load documents from file or use sample documents.
        
        Args:
            file_path: Path to text file or PDF
            sample_docs: List of sample document strings
            
        Returns:
            List of Document objects
        """
        docs = []
        
        if sample_docs:
            # Use provided sample documents
            for i, content in enumerate(sample_docs):
                doc = Document(
                    page_content=content,
                    metadata={"source": "sample", "doc_id": i}
                )
                docs.append(doc)
        elif file_path:
            # Load from file
            try:
                if file_path.lower().endswith('.pdf'):
                    docs = DocumentLoader.load_pdf(file_path)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Split by double newlines to create documents
                        chunks = content.split('\n\n')
                        for i, chunk in enumerate(chunks):
                            if chunk.strip():
                                doc = Document(
                                    page_content=chunk.strip(),
                                    metadata={"source": file_path, "doc_id": i}
                                )
                                docs.append(doc)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
        
        return docs
    
    @staticmethod
    def hash_text(text: str) -> str:
        """
        Generate hash of text for deduplication.
        
        Args:
            text: Text to hash
            
        Returns:
            SHA256 hash
        """
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    
    @staticmethod
    def attach_structure(elements: List[Document], source: str) -> List[Document]:
        """
        Attach structural metadata to documents based on sections.
        
        Args:
            elements: List of Document objects
            source: Source identifier
            
        Returns:
            List of Document objects with enriched metadata
        """
        structured = []
        current_section = "General"
        
        for el in elements:
            category = el.metadata.get("category")
            
            if category in ["Title", "Header"]:
                current_section = el.page_content.strip()
            
            el.metadata.update({
                "source": source,
                "section": current_section,
                "page": el.metadata.get("page_number"),
                "content_type": "table" if category == "Table" else "text",
                "chunk_hash": DocumentLoader.hash_text(el.page_content)
            })
            structured.append(el)
        
        return structured
    
    @staticmethod
    def chunk_documents(documents: List[Document], chunk_size: int = 500) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk in characters
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata
            
            # Split by sentences or fixed chunk size
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                if chunk.strip():
                    new_doc = Document(
                        page_content=chunk,
                        metadata={**metadata, "chunk_id": i // chunk_size}
                    )
                    chunked_docs.append(new_doc)
        
        return chunked_docs


class FinancialIndustryChunker:
    """
    Specialized chunking for financial documents with hierarchical parent-child chunks
    """
    
    # Keywords indicating financial clauses
    CLAUSE_KEYWORDS = [
        "provided that",
        "subject to",
        "notwithstanding",
        "except where",
        "unless otherwise",
        "in accordance with"
    ]
    
    # Pattern for numeric and currency content
    NUMERIC_PATTERN = re.compile(
        r"(₹|\$|%|\d+\s?(?:percent|years|months|days))",
        re.IGNORECASE
    )
    
    def __init__(self, parent_chunk_size: int = 1800, child_chunk_size: int = 700):
        """
        Initialize financial chunker.
        
        Args:
            parent_chunk_size: Size for parent chunks (context)
            child_chunk_size: Size for child chunks (vector embeddings)
        """
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " "]
        )
        
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=0
        )
    
    @staticmethod
    def contains_clause(text: str) -> bool:
        """Check if text contains financial clauses."""
        t = text.lower()
        return any(k in t for k in FinancialIndustryChunker.CLAUSE_KEYWORDS)
    
    @staticmethod
    def contains_numeric(text: str) -> bool:
        """Check if text contains numeric or currency values."""
        return bool(FinancialIndustryChunker.NUMERIC_PATTERN.search(text))
    
    @staticmethod
    def text_hash(text: str) -> str:
        """Generate hash for deduplication."""
        return hashlib.sha256(text.strip().encode()).hexdigest()
    
    def chunk(self, structured_docs: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Split documents into parent-child chunks optimized for financial content.
        
        Args:
            structured_docs: Pre-structured documents with metadata
            
        Returns:
            Tuple of (vector_chunks, parent_chunks)
        """
        parent_chunks = []
        vector_chunks = []
        seen_hashes = set()
        
        for doc in structured_docs:
            content_type = doc.metadata.get("content_type", "text")
            text = doc.page_content
            
            # Handle atomic content (tables, formulas, clauses with numbers)
            if (
                content_type in ["table", "formula", "list"]
                or self.contains_clause(text)
                or self.contains_numeric(text)
            ):
                h = self.text_hash(text)
                if h in seen_hashes:
                    continue
                
                parent_id = str(uuid.uuid4())
                doc.metadata.update({
                    "parent_id": parent_id,
                    "granularity": "atomic",
                    "contains_clauses": self.contains_clause(text),
                    "contains_numeric": self.contains_numeric(text)
                })
                
                parent_chunks.append(doc)
                vector_chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            "parent_id": parent_id,
                            "section": doc.metadata.get("section"),
                            "page": doc.metadata.get("page"),
                            "source": doc.metadata.get("source"),
                            "content_type": content_type,
                            "granularity": "atomic"
                        }
                    )
                )
                seen_hashes.add(h)
                continue
            
            # Create parent chunks for context
            parents = self.parent_splitter.split_documents([doc])
            
            for parent in parents:
                parent_id = str(uuid.uuid4())
                parent.metadata.update({
                    "parent_id": parent_id,
                    "granularity": "parent"
                })
                parent_chunks.append(parent)
                
                # Create child chunks for embeddings
                children = self.child_splitter.split_documents([parent])
                for child in children:
                    h = self.text_hash(child.page_content)
                    if h in seen_hashes:
                        continue
                    
                    vector_chunks.append(
                        Document(
                            page_content=child.page_content,
                            metadata={
                                "parent_id": parent_id,
                                "section": child.metadata.get("section"),
                                "page": child.metadata.get("page"),
                                "source": child.metadata.get("source"),
                                "content_type": content_type,
                                "granularity": "child",
                                "contains_numeric": self.contains_numeric(child.page_content)
                            }
                        )
                    )
                    seen_hashes.add(h)
        
        return vector_chunks, parent_chunks
