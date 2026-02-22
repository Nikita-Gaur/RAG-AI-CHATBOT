
import json
from typing import List, Dict, Optional
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retriever import Retriever


class RAGPipeline:
    """Main RAG pipeline combining retrieval and generation"""
    
    def __init__(self, retriever: Retriever, llm_model: str = "mistral", temperature: float = 0.7, use_cache: bool = True):
        """
        Initialize RAG pipeline.
        
        Args:
            retriever: Retriever instance for document retrieval
            llm_model: Ollama model name (default: mistral)
                      Options: mistral, llama2, neural-chat, orca-mini
            temperature: LLM temperature for response generation
            use_cache: Whether to use conversation caching (default: True)
        """
        self.retriever = retriever
        
        # Initialize conversation cache (optional)
        # WHY? To store conversation history across sessions
        # Last 5 conversations are automatically kept
        # No conversation cache
        self.cache = None
        self.current_conversation_id: Optional[str] = None
        
        print("    Loading Ollama language model...")
        # Use Ollama for local inference with better quality answers
        try:
            self.llm = OllamaLLM(
                model=llm_model,  # mistral, llama2, neural-chat, etc.
                temperature=temperature,
                base_url="http://localhost:11434",  # Default Ollama port
                # NEW: Optimize for detailed, comprehensive answers
                top_k=40,        # Control token diversity
                top_p=0.9,       # Nucleus sampling
                num_predict=512  # Allow up to 512 tokens (~800 words)
            )
            print(f"✓ Ollama LLM '{llm_model}' loaded successfully")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            print("Using enhanced template-based responses instead")
            self.llm = None
        
        # Create prompt template for financial documents
        # OPTIMIZED: Ask for detailed, comprehensive, well-structured answers
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are an expert financial advisor assistant. Your task is to provide detailed, comprehensive, and well-structured answers.

Context Documents:
{context}

User Question: {question}

Instructions:
1. Provide a thorough and detailed answer (at least 3-5 paragraphs)
2. Include specific details, numbers, and references from the context
3. Structure your answer with clear sections if applicable
4. Explain concepts in depth, not just brief mentions
5. If multiple related points exist, cover them all comprehensively
6. Use bullet points for lists when appropriate
7. End with a brief summary of key takeaways

Detailed Answer:"""
        )
    
    def start_new_conversation(self, conversation_id: str = None) -> str:
        """
        Start a new conversation session.
        
        WHY?
        - Creates new UUID for chat
        - Allows switching between conversations
        
        Args:
            conversation_id: Optional custom ID
            
        Returns:
            Conversation ID
        """
        import uuid
        if conversation_id is None:
            conversation_id = str(uuid.uuid4())[:8]
        self.current_conversation_id = conversation_id
        return self.current_conversation_id
    
    def load_conversation(self, conversation_id: str) -> bool:
        """
        Load existing conversation. (No-op, always returns False)
        """
        print("No conversation cache available. Cannot load conversation.")
        return False
        
        # Create prompt template for financial documents
        self.prompt_template = ChatPromptTemplate.from_template(
            """You are a helpful financial advisor assistant. Use the following context to answer the question accurately.
            
Context:
{context}

Question: {question}

Please provide a clear, accurate answer based on the provided context. If the context doesn't contain relevant information, please state that."""
        )
        
    def run(self, query: str, k: int = 3) -> dict:
        """
        Run the RAG pipeline with cache optimization.
        
        OPTIMIZED FLOW:
        1. Check cache FIRST for exact matching question
        2. If found → Return cached answer (INSTANT - <100ms)
        3. If NOT found → Run full RAG pipeline (2-4 seconds)
        4. Store new Q&A in cache for future use
        
        WHY THIS IS BETTER?
        - Same questions return instantly from cache
        - No document retrieval needed for cached answers
        - No LLM inference needed for cached answers
        - Dramatically faster for repeated questions
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        # OPTIMIZATION: Check cache FIRST
        # WHY? Return cached answer immediately if question was asked before
        if self.current_conversation_id:
            history = self.get_conversation_history()
            
            # Search for exact matching question in history (case-insensitive)
            for i, msg in enumerate(history):
                if (msg['role'] == 'user' and 
                    msg['content'].strip().lower() == query.strip().lower()):
                    
                    # Found exact match! Get the assistant's response
                    if i + 1 < len(history) and history[i + 1]['role'] == 'assistant':
                        cached_answer = history[i + 1]['content']
                        cached_suggestions = history[i + 1].get('suggestions', [])  # Get cached suggestions
                        print("⚡ Retrieved from cache (instant!)")
                        return {
                            "query": query,
                            "answer": cached_answer,
                            "suggested_questions": cached_suggestions,  # Use cached suggestions (no LLM call!)
                            "retrieved_chunks": [],
                            "context_docs": [],
                            "sources": [],
                            "chunk_count": 0,
                            "context_count": 0,
                            "conversation_id": self.current_conversation_id,
                            "from_cache": True,  # Flag: came from cache
                            "suggestions_cached": True  # Flag: suggestions from cache, not regenerated
                        }
        
        # NOT in cache → Run full RAG pipeline
        # Step 0: Add user query to cache
        # WHY? Save conversation history for later retrieval
        # No cache: skip saving user query
        
        # Step 1: Retrieve relevant documents with context
        retrieval_result = self.retriever.retrieve_with_context(query, k=k)
        chunks = retrieval_result["chunks"]
        context_docs = retrieval_result["context"]
        
        # Step 2: Format context from retrieved documents
        context = self._format_context(context_docs)
        
        # Step 3: Generate answer using LLM
        if self.llm:
            try:
                # For Ollama, create a detailed prompt for comprehensive answers
                prompt = f"""You are an expert financial advisor assistant. Provide detailed, comprehensive, well-structured answers.

Context Documents:
{context}

User Question: {query}

Instructions:
1. Provide a thorough answer (at least 3-5 paragraphs)
2. Include specific details, numbers, and references from the context
3. Structure your answer with clear sections if applicable
4. Explain concepts in depth
5. If multiple related points exist, cover them all
6. Use bullet points for lists when appropriate
7. End with key takeaways

Detailed Answer:"""
                
                # Generate response using Ollama
                answer = self.llm.invoke(prompt)
            except Exception as e:
                print(f"LLM Error: {e}")
                answer = self._generate_summary_answer(context, query)
        else:
            answer = self._generate_summary_answer(context, query)
        
        # Step 4: Add assistant response to cache
        # WHY? Complete the Q&A pair in conversation history for future use
        # No cache: skip saving assistant response
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [doc.page_content for doc in chunks],
            "context_docs": context_docs,
            "sources": [doc.metadata for doc in chunks],
            "chunk_count": len(chunks),
            "context_count": len(context_docs),
            "conversation_id": self.current_conversation_id,
            "from_cache": False,
            "suggested_questions": []
        }
    
    def _generate_summary_answer(self, context: str, query: str) -> str:
        """
        Generate a detailed answer by extracting and summarizing context.
        
        IMPROVED: Returns comprehensive answers
        - Extracts multiple substantial paragraphs
        - Includes detailed information
        - Preserves context structure
        - Fallback when LLM fails to load
        - Strips out any accidentally included suggestions
        
        Args:
            context: Retrieved context text
            query: User query
            
        Returns:
            Generated answer (comprehensive)
        """
        # Split context into paragraphs
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        # Return all substantial paragraphs (not limited to 2)
        if paragraphs:
            selected_paragraphs = []
            for para in paragraphs:
                # Include all paragraphs > 50 chars
                if len(para) > 50:
                    selected_paragraphs.append(para)
            
            if selected_paragraphs:
                # Return ALL selected paragraphs (not limited)
                return "\n\n".join(selected_paragraphs)
        
        # Fallback: return more context (600 chars instead of 400)
        if context.strip():
            answer = context[:800].strip()
            # Add "..." only if we're cutting off mid-sentence
            if len(context) > 800:
                answer += "\n\n[Additional content available - this is a summary]"
            return answer
        else:
            return "Unable to generate answer from available documents."
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            section = doc.metadata.get("section", "General")
            page = doc.metadata.get("page", "N/A")
            content_type = doc.metadata.get("content_type", "text")
            
            context_parts.append(
                f"[Document {i} | Section: {section} | Page: {page} | Type: {content_type}]\n"
                f"{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    def batch_run(self, queries: List[str], k: int = 3) -> List[dict]:
        """
        Run RAG pipeline on multiple queries.
        
        Args:
            queries: List of queries
            k: Number of documents to retrieve per query
            
        Returns:
            List of results
        """
        results = []
        for query in queries:
            result = self.run(query, k=k)
            results.append(result)
        
        return results
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get all messages from current conversation. (No-op, returns empty list)
        """
        return []
    
    def get_all_conversations(self) -> List[Dict]:
        """
        Get all 5 stored conversations. (No-op, returns empty list)
        """
        return []
    
    def get_conversation_summary(self) -> Dict:
        """
        Get summary of all conversations. (No-op, returns empty summary)
        """
        return {"total_conversations": 0, "conversations": []}
