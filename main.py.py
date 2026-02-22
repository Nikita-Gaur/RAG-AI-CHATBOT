"""
Main example script demonstrating the RAG pipeline with financial document chunking
"""
import os
import sys
from io import StringIO
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import DocumentLoader, FinancialIndustryChunker
from retriever import SimpleVectorStore, Retriever
from rag_pipeline import RAGPipeline
from follow_up_generator import FollowUpGenerator


def display_response(response: dict):
    """Display answer + follow-up questions with confidence level"""
    
    # Show understanding
    print(f"\n🔍 Understanding: {response.get('understanding', 'N/A')}")
    
    # Show confidence
    confidence = response.get('confidence', 'low')
    if confidence.lower() == 'high':
        print("📊 Confidence: ✅ HIGH")
    elif confidence.lower() == 'medium':
        print("📊 Confidence: ⚠️ MEDIUM")
    else:
        print("📊 Confidence: ❓ LOW")
    
    # Show answer
    print(f"\n🤖 Answer:\n{response.get('answer', 'No answer generated')}")
    
    # Show follow-up questions
    questions = response.get('follow_up_questions', [])[:3]  # ENFORCE: Max 3 questions
    if questions:
        print("\n💡 You might also want to ask:")
        for i, q in enumerate(questions, 1):
            print(f"  {i}. {q}")
    
    print()


def main_with_pdf(pdf_path: str):
    """Run RAG pipeline with PDF documents"""
    
    # Load environment variables
    load_dotenv()
    
    # Extract data folder from PDF path
    data_folder = os.path.dirname(pdf_path)
    cache_dir = os.path.join(data_folder, "embeddings_cache")
    
    # Step 1: Load PDF
    loader = DocumentLoader()
    try:
        raw_docs = loader.load_pdf(pdf_path)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return
    
    # Step 2: Attach structural metadata
    structured_docs = DocumentLoader.attach_structure(raw_docs, source="PDF_Document")
    
    # Step 3: Apply financial industry chunking
    chunker = FinancialIndustryChunker(parent_chunk_size=1800, child_chunk_size=700)
    vector_chunks, parent_chunks = chunker.chunk(structured_docs)
    
    # Step 4: Create embeddings and vector store with caching
    print("🔧 Setting up vector store...")
    
    # Suppress verbose initialization output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = SimpleVectorStore(embeddings, cache_dir=cache_dir)
    vector_store.add_documents(vector_chunks, parent_documents=parent_chunks)
    
    # Restore stdout
    sys.stdout = old_stdout
    
    print("✓ Vector store ready")
    
    # Step 5: Initialize retriever
    retriever = Retriever(vector_store, expand_context=True)
    
    # Step 6: Initialize RAG pipeline (suppress verbose output)
    print("🚀 Loading LLM...")
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    rag = RAGPipeline(retriever, temperature=0.7, use_cache=False)
    sys.stdout = old_stdout
    print("✓ Ready!\n")
    
    # Initialize follow-up questions generator
    followup_generator = FollowUpGenerator(rag.llm)
    
    # Step 7: Interactive query loop
    print("\n" + "=" * 70)
    print("💬 ASK QUESTIONS ABOUT THE DOCUMENT")
    print("=" * 70)
    print("Type 'quit' to exit\n")
    
    query_count = 0
    while True:
        user_query = input("📝 Your question: ").strip()
        
        if user_query.lower() == "quit":
            print("\n✓ Goodbye!")
            break
        
        if not user_query:
            print("⚠ Please enter a question.\n")
            continue
        
        query_count += 1
        print(f"⏳ Processing...", end='', flush=True)
        
        try:
            # Get answer from RAG
            result = rag.run(user_query, k=2)
            context_docs = result.get('context_docs', [])
            
            # Generate answer + 4-5 follow-up questions
            response = followup_generator.generate(user_query, context_docs)
            
            # Display nicely
            display_response(response)
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    # Run with PDF from your data folder
    pdf_path = r"C:\Users\anandsagar\OneDrive - Deloitte (O365D)\Pictures\JPMC\Practice\practice_2\data\Financialadvisorsroleininfluencingdecisions.pdf"
    main_with_pdf(pdf_path)