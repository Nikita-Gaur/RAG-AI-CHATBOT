"""
Interactive RAG chatbot
Stores conversation history

PERFORMANCE NOTES:
- First query: ~2-4 sec (includes LLM + embeddings)
"""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from document_loader import DocumentLoader, FinancialIndustryChunker
from retriever import SimpleVectorStore, Retriever
from rag_pipeline import RAGPipeline
from follow_up_generator import FollowUpGenerator

load_dotenv()


def print_banner():
    """Display welcome banner"""
    print("\n" + "="*70)
    print("RAG CHATBOT")
    print("="*70)
    print("\n📋 COMMANDS:")
    print("  'new'         - Start new conversation")
    print("  'history'     - Show current conversation history")
    print("  'list'        - Show all 5 stored conversations")
    print("  'load <id>'   - Load previous conversation")
    print("  'clear'       - Delete current conversation")
    print("  'summary'     - Show conversation summary")
    print("  'quit'        - Exit")
    print("="*70 + "\n")


def initialize_rag():
    """Initialize RAG pipeline with embeddings"""
    print("[1] Loading PDF...")
    pdf_path = r"C:\Users\anandsagar\OneDrive - Deloitte (O365D)\Pictures\JPMC\Practice\practice_2\data\Financialadvisorsroleininfluencingdecisions.pdf"
    
    loader = DocumentLoader()
    raw_docs = loader.load_pdf(pdf_path)
    print(f"    ✓ Loaded {len(raw_docs)} pages")
    
    print("[2] Processing documents...")
    structured_docs = DocumentLoader.attach_structure(raw_docs, source="PDF_Document")
    chunker = FinancialIndustryChunker()
    vector_chunks, parent_chunks = chunker.chunk(structured_docs)
    print(f"    ✓ Created {len(vector_chunks)} vector chunks")
    
    print("[3] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = SimpleVectorStore(embeddings, cache_dir="./embeddings_cache")
    vector_store.add_documents(vector_chunks, parent_documents=parent_chunks)
    
    print("[4] Initializing RAG...")
    retriever = Retriever(vector_store, expand_context=True)
    # USING MISTRAL: More accurate and detailed answers
    rag = RAGPipeline(retriever, llm_model="mistral", use_cache=False)
    
    # Initialize follow-up question generator
    followup_generator = FollowUpGenerator(rag.llm)
    rag.followup_generator = followup_generator
    
    return rag


def show_history(rag):
    """
    Display current conversation history.
    
    WHY THIS FUNCTION?
    - Users want to see past messages
    - Track conversation flow
    - Verify what was discussed
    """
    history = rag.get_conversation_history()
    
    if not history:
        print("📝 No messages in current conversation")
        return
    
    print("\n📜 CONVERSATION HISTORY:")
    print("-" * 70)
    for i, msg in enumerate(history, 1):
        role = "👤 You" if msg['role'] == "user" else "🤖 Bot"
        timestamp = msg.get('timestamp', 'N/A')
        content = msg['content']
        print(f"{i}. {role} ({timestamp}):")
        print(f"   {content}")
    print("-" * 70 + "\n")


def show_all_conversations(rag):
    """
    Display all 5 stored conversations.
    
    WHY THIS FUNCTION?
    - Users switch between chats
    - See which conversations are saved
    - Pick one to load
    """
    conversations = rag.get_all_conversations()
    
    if not conversations:
        print("📚 No stored conversations yet")
        return
    
    print("\n📚 ALL CONVERSATIONS (Last 5):")
    print("-" * 70)
    for i, conv in enumerate(conversations, 1):
        conv_id = conv['conversation_id']
        msg_count = len(conv.get('messages', []))
        created = conv['created_at'][:16]  # Show date and time only
        
        print(f"{i}. ID: {conv_id[:8]}...")
        print(f"   Messages: {msg_count}")
        print(f"   Created: {created}")
    print("-" * 70 + "\n")


def show_summary(rag):
    """
    Display summary of all conversations.
    
    WHY THIS FUNCTION?
    - Quick overview without full details
    - Useful for debugging
    """
    summary = rag.get_conversation_summary()
    print("\n📊 CONVERSATION SUMMARY:")
    print("-" * 70)
    print(f"Total Conversations: {summary['total_conversations']}")
    for conv in summary['conversations']:
        print(f"  • {conv['id']} - {conv['message_count']} messages")
    print("-" * 70 + "\n")


def main():
    """Main chat loop"""
    print("\n⏳ Initializing RAG Pipeline...")
    rag = initialize_rag()
    
    print_banner()
    
    # Start new conversation
    conv_id = rag.start_new_conversation()
    print(f"✓ Started new conversation: {conv_id[:8]}...\n")
    
    # Main chat loop
    while True:
        user_input = input("\n📝 You: ").strip()
        
        if not user_input:
            continue
        
        # COMMAND: Quit
        if user_input.lower() == "quit":
            print("\n👋 Goodbye!")
            break
        
        # COMMAND: New conversation
        elif user_input.lower() == "new":
            conv_id = rag.start_new_conversation()
            print(f"✓ New conversation: {conv_id[:8]}...\n")
            last_suggestions = []  # Clear suggestions on new conversation
            continue
        
        # COMMAND: Show history
        elif user_input.lower() == "history":
            show_history(rag)
            continue
        
        # COMMAND: List all conversations
        elif user_input.lower() == "list":
            show_all_conversations(rag)
            continue
        
        # COMMAND: Load conversation
        elif user_input.lower().startswith("load"):
            parts = user_input.split()
            if len(parts) < 2:
                print("Usage: load <conversation_id>")
                continue
            
            conv_id = parts[1]
            # Try to load with partial ID
            conversations = rag.get_all_conversations()
            found = False
            for conv in conversations:
                if conv['conversation_id'].startswith(conv_id):
                    if rag.load_conversation(conv['conversation_id']):
                        found = True
                    break
            
            if not found:
                print(f"✗ Conversation not found: {conv_id}")
            print()
            last_suggestions = []  # Clear suggestions when loading conversation
            continue
        
        # COMMAND: Clear current conversation
        elif user_input.lower() == "clear":
            if rag.current_conversation_id:
                rag.current_conversation_id = rag.start_new_conversation()
                print("✓ Conversation cleared, started new one\n")
            continue
        
        # COMMAND: Show summary
        elif user_input.lower() == "summary":
            show_summary(rag)
            continue
        
        # NORMAL QUERY: Process with RAG
        else:
            # Confirm full question was received
            print(f"\n✓ Question: {user_input}")
            print("⏳ Processing...")
            try:
                result = rag.run(user_input, k=2)
                print(f"\n🤖 Bot: {result['answer']}")
                
                # Generate follow-up questions
                try:
                    if hasattr(rag, 'followup_generator') and rag.llm:
                        context_docs = result.get('context_docs', [])
                        followup_result = rag.followup_generator.generate(user_input, context_docs)
                        if followup_result and 'follow_up_questions' in followup_result:
                            # ENFORCE: Always return exactly 3 questions
                            last_suggestions = followup_result['follow_up_questions'][:3]
                            print("\n💡 Suggested follow-up questions:")
                            for i, q in enumerate(last_suggestions, 1):
                                print(f"   {i}. {q}")
                except Exception as e:
                    print(f"   (Could not generate suggestions: {str(e)})")
                
                print(f"\n📊 [Chunks retrieved: {result['chunk_count']}, Conversation ID: {result['conversation_id'][:8]}...]\n")
            except Exception as e:
                print(f"\n✗ Error: {e}\n")


if __name__ == "__main__":
    main()
