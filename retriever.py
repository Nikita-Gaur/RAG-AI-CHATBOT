"""
Vector store and retrieval module for RAG pipeline
"""
from typing import List, Dict, Optional
import numpy as np
import os
import json
import pickle
from pathlib import Path
from langchain_core.documents import Document


class SimpleVectorStore:
    """Simple vector store using FAISS for semantic search with persistence"""
    
    def __init__(self, embeddings, cache_dir: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            embeddings: LangChain embeddings object
            cache_dir: Directory to cache embeddings and index (optional)
        """
        self.embeddings = embeddings
        self.documents = []
        self.vectors = None
        self.index = None
        self.parent_docs = []  # Store parent chunks for context
        self.cache_dir = cache_dir
        
        # Create cache directory if provided
        if self.cache_dir:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
    def add_documents(self, documents: List[Document], parent_documents: Optional[List[Document]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects (vector chunks)
            parent_documents: Optional list of parent Document objects for context
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        self.documents = documents
        self.parent_docs = parent_documents or []
        
        # Check if embeddings are cached
        if self.cache_dir and self._cache_exists():
            print("📦 Loading embeddings from cache...")
            self._load_from_cache()
            return
        
        print("🔄 Generating embeddings...")
        # Generate embeddings for all documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        self.vectors = embeddings_array
        
        print(f"✓ Added {len(documents)} vector chunks to store")
        if parent_documents:
            print(f"✓ Added {len(parent_documents)} parent chunks for context")
        
        # Save to cache if directory provided
        if self.cache_dir:
            print("💾 Saving embeddings to cache...")
            self._save_to_cache()
            print("✓ Cache saved successfully")
    
    def _cache_exists(self) -> bool:
        """Check if cache files exist."""
        if not self.cache_dir:
            return False
        
        index_path = os.path.join(self.cache_dir, "index.faiss")
        vectors_path = os.path.join(self.cache_dir, "vectors.npy")
        docs_path = os.path.join(self.cache_dir, "documents.pkl")
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        
        return (os.path.exists(index_path) and 
                os.path.exists(vectors_path) and 
                os.path.exists(docs_path) and
                os.path.exists(metadata_path))
    
    def _save_to_cache(self):
        """Save embeddings, index, and documents to cache."""
        if not self.cache_dir or not self.index:
            return
        
        try:
            import faiss
            
            # Save FAISS index
            index_path = os.path.join(self.cache_dir, "index.faiss")
            faiss.write_index(self.index, index_path)
            
            # Save vectors
            vectors_path = os.path.join(self.cache_dir, "vectors.npy")
            np.save(vectors_path, self.vectors)
            
            # Save documents (pickled)
            docs_path = os.path.join(self.cache_dir, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save parent documents
            parent_docs_path = os.path.join(self.cache_dir, "parent_documents.pkl")
            with open(parent_docs_path, 'wb') as f:
                pickle.dump(self.parent_docs, f)
            
            # Save metadata
            metadata = {
                "num_documents": len(self.documents),
                "num_parent_documents": len(self.parent_docs),
                "vector_dimension": self.vectors.shape[1] if self.vectors is not None else 0,
                "index_type": "IndexFlatL2"
            }
            metadata_path = os.path.join(self.cache_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _load_from_cache(self):
        """Load embeddings, index, and documents from cache."""
        if not self.cache_dir:
            return
        
        try:
            import faiss
            
            # Load FAISS index
            index_path = os.path.join(self.cache_dir, "index.faiss")
            self.index = faiss.read_index(index_path)
            
            # Load vectors
            vectors_path = os.path.join(self.cache_dir, "vectors.npy")
            self.vectors = np.load(vectors_path)
            
            # Load documents
            docs_path = os.path.join(self.cache_dir, "documents.pkl")
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load parent documents
            parent_docs_path = os.path.join(self.cache_dir, "parent_documents.pkl")
            if os.path.exists(parent_docs_path):
                with open(parent_docs_path, 'rb') as f:
                    self.parent_docs = pickle.load(f)
            
            print(f"✓ Loaded {len(self.documents)} vector chunks from cache")
            print(f"✓ Loaded {len(self.parent_docs)} parent chunks from cache")
                
        except Exception as e:
            print(f"Warning: Could not load cache: {e}")
            raise
    
    def clear_cache(self):
        """Clear cached embeddings and index."""
        if not self.cache_dir:
            return
        
        try:
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
                print(f"✓ Cache cleared: {self.cache_dir}")
        except Exception as e:
            print(f"Warning: Could not clear cache: {e}")
        
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of similar Document objects
        """
        if self.index is None:
            raise ValueError("No documents in vector store. Call add_documents first.")
        
        # Generate embedding for query
        query_embedding = self.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Return documents
        results = []
        for idx in indices[0]:
            if idx < len(self.documents):
                results.append(self.documents[idx])
        
        return results
    
    def get_parent_chunk(self, parent_id: str) -> Optional[Document]:
        """
        Retrieve parent chunk by ID for context expansion.
        
        Args:
            parent_id: Parent chunk ID
            
        Returns:
            Parent Document or None
        """
        for doc in self.parent_docs:
            if doc.metadata.get("parent_id") == parent_id:
                return doc
        return None


class Retriever:
    """Wrapper for retrieval with metadata filtering and context expansion"""
    
    def __init__(self, vector_store: SimpleVectorStore, expand_context: bool = True):
        """
        Initialize retriever.
        
        Args:
            vector_store: SimpleVectorStore instance
            expand_context: Whether to include parent chunks for context
        """
        self.vector_store = vector_store
        self.expand_context = expand_context
        
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of relevant Document objects
        """
        return self.vector_store.similarity_search(query, k=k)
    
    def retrieve_with_context(self, query: str, k: int = 3) -> Dict[str, List[Document]]:
        """
        Retrieve documents with parent context expansion.
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with 'chunks' and 'context' keys
        """
        # Get most relevant chunks
        chunks = self.retrieve(query, k=k)
        
        context_docs = []
        seen_parent_ids = set()
        
        # Expand with parent chunks for better context
        if self.expand_context:
            for chunk in chunks:
                parent_id = chunk.metadata.get("parent_id")
                if parent_id and parent_id not in seen_parent_ids:
                    parent = self.vector_store.get_parent_chunk(parent_id)
                    if parent:
                        context_docs.append(parent)
                        seen_parent_ids.add(parent_id)
        
        return {
            "chunks": chunks,
            "context": context_docs if context_docs else chunks
        }
