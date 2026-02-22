"""Quick test to verify exactly 3 follow-up questions are returned"""
import sys
from io import StringIO
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import DocumentLoader, FinancialIndustryChunker
from retriever import SimpleVectorStore, Retriever
from rag_pipeline import RAGPipeline
from follow_up_generator import FollowUpGenerator

load_dotenv()

# Initialize RAG
print("[1] Loading PDF...")
pdf_path = r"C:\Users\anandsagar\OneDrive - Deloitte (O365D)\Pictures\JPMC\Practice\practice_2\data\Financialadvisorsroleininfluencingdecisions.pdf"

loader = DocumentLoader()
raw_docs = loader.load_pdf(pdf_path)

print("[2] Processing documents...")
structured_docs = DocumentLoader.attach_structure(raw_docs, source="PDF_Document")
chunker = FinancialIndustryChunker()
vector_chunks, parent_chunks = chunker.chunk(structured_docs)

print("[3] Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = SimpleVectorStore(embeddings, cache_dir="./embeddings_cache")
vector_store.add_documents(vector_chunks, parent_documents=parent_chunks)

print("[4] Initializing RAG...")
retriever = Retriever(vector_store, expand_context=True)
rag = RAGPipeline(retriever, llm_model="mistral", use_cache=False)

followup_generator = FollowUpGenerator(rag.llm)
rag.followup_generator = followup_generator

# Test query
user_query = "How many respondents neither agree nor disagree with the capability of financial advisors to handle a financial crisis?"
print(f"\n{'='*70}")
print(f"QUERY: {user_query}")
print(f"{'='*70}\n")

# Run RAG
result = rag.run(user_query, k=2)
print(f"\n🤖 ANSWER:\n{result['answer']}\n")

# Generate follow-ups
context_docs = result.get('context_docs', [])
followup_result = rag.followup_generator.generate(user_query, context_docs)

# Display results
if followup_result and 'follow_up_questions' in followup_result:
    questions = followup_result['follow_up_questions'][:3]  # ENFORCE: Max 3 questions
    print(f"\n💡 FOLLOW-UP QUESTIONS ({len(questions)}):")
    for i, q in enumerate(questions, 1):
        print(f"   {i}. {q}")
else:
    print("No follow-up questions generated")
