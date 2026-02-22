## Key Concepts & Tips

- **RAG (Retrieval-Augmented Generation):** Combines document retrieval with generative AI to provide grounded, context-aware answers.
- **Embeddings:** Numeric representations of text used for semantic search. Cached for speed.
- **FAISS:** Library for efficient similarity search on embeddings.
- **Ollama:** Local LLM server. Ensure it is running (`ollama serve`) before starting the system.
- **Chunking:** Documents are split into overlapping chunks for better retrieval granularity.
- **Caching:** Both embeddings and Q&A pairs are cached for performance.

### Usage Tips
- For best results, use clear, specific questions.
- If answers seem slow, check Ollama status and consider switching to a faster model (see PERFORMANCE_GUIDE.py).
- To refresh embeddings, delete the `embeddings_cache/` folder and rerun the pipeline.
- Adjust the number of context chunks (`k`) for a balance between speed and answer quality.

### Troubleshooting
- **Ollama connection errors:** Make sure Ollama is running and accessible at `http://localhost:11434`.
- **No answers returned:** Check that documents are loaded and embeddings are generated.
- **Performance issues:** See PERFORMANCE_GUIDE.py for optimization strategies.

# Project Overview

This file contains all essential information related to the cache_based_sys project.


## Architecture Overview

### High-Level Architecture Diagram (Textual)

```
User Query
   |
   v
[Chat/CLI Interface]
   |
   v
[RAGPipeline]
   |      |         |
   |      |         +---> [FollowUpGenerator]
   |      |
   |      +---> [Retriever]
   |                |
   |                +---> [SimpleVectorStore (FAISS)]
   |                              |
   |                              +---> [Embeddings Cache (index.faiss, vectors.npy, metadata.json)]
   |
   +---> [LLM (Ollama, e.g., mistral/gemma3)]
   |
   v
Answer + Follow-up Questions
```

### Component Descriptions

- **Chat/CLI Interface**: Handles user input and displays answers/follow-ups.
- **RAGPipeline**: Orchestrates retrieval, context formatting, LLM invocation, and caching.
- **Retriever**: Finds relevant document chunks using semantic search.
- **SimpleVectorStore**: Stores and retrieves document embeddings using FAISS for fast similarity search.
- **Embeddings Cache**: Stores precomputed embeddings and metadata for fast startup and repeated queries.
- **LLM (Ollama)**: Generates detailed answers and follow-up questions based on retrieved context.
- **FollowUpGenerator**: Suggests follow-up questions grounded in the context.
- **DocumentLoader & FinancialIndustryChunker**: Loads and splits documents into hierarchical chunks for better retrieval.

### System Workflow

1. **Document Loading & Chunking**
   - Documents (PDF/text) are loaded and split into parent/child chunks.
   - Embeddings are generated for each chunk and stored in the FAISS vector store (with metadata).

2. **User Query Handling**
   - User submits a question via chat or CLI.
   - The RAGPipeline checks the cache for a previous answer.
   - If not cached, the Retriever finds the most relevant chunks.
   - The context is formatted and sent to the LLM (Ollama) for answer generation.
   - The FollowUpGenerator proposes follow-up questions based on the answer/context.
   - The answer and follow-ups are returned to the user and optionally cached for future queries.

3. **Performance Optimizations**
   - Embeddings are cached for fast repeated runs.
   - LLM model can be switched for speed (e.g., gemma3 for faster responses).
   - Number of context chunks (k) can be tuned for speed/accuracy.
   - Parallel processing and hardware acceleration are possible future improvements.

---

## Project Structure

- **analyze_followup_source.py**: Analyze follow-up sources.
- **chat.py**: Chat interface or logic.
- **check_followup_logic.py**: Logic for checking follow-ups.
- **debug_followups.py**: Debugging follow-up logic.
- **document_loader.py**: Loads documents for processing.
- **follow_up_generator.py**: Generates follow-up questions or actions.
- **main.py**: Main entry point for the project.
- **PERFORMANCE_GUIDE.py**: Performance guidelines and tips.
- **quick_test.py**: Quick tests for the system.
- **rag_pipeline.py**: Retrieval-Augmented Generation pipeline.
- **requirements.txt**: Python dependencies.
- **retriever.py**: Retrieval logic for documents or data.
- **test_3_questions.py**: Test script for three questions.
- **test_all_questions.py**: Test script for all questions.
- **test_improved_prompt.py**: Test script for improved prompts.
- **test_prompt_format.py**: Test script for prompt formatting.
- **embeddings_cache/**: Stores cached embeddings (index.faiss, metadata.json, vectors.npy).

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main.py
   ```

## Description

This project implements a cache-based system for document retrieval and follow-up question generation, likely using RAG (Retrieval-Augmented Generation) techniques. It includes utilities for loading documents, generating and debugging follow-ups, and running various tests.

## Folder Descriptions

- **embeddings_cache/**: Contains FAISS index and related files for fast vector search.
- **__pycache__/**: Python bytecode cache (auto-generated).

## Performance Guide

See PERFORMANCE_GUIDE.py for tips on optimizing the system.

## Authors
- [Your Name Here]

## Last Updated
- February 22, 2026

---

Add more details as the project evolves.
