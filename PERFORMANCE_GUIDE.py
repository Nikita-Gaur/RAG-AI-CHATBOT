"""
PERFORMANCE OPTIMIZATION GUIDE
===============================

Current Performance Profile:
- Embeddings (first time): 16.39s (one-time cost, then cached)
- Ollama mistral: ~2-5s per query
- Ollama gemma3: ~1-2s per query
- Retrieval + formatting: ~0.5s
- TOTAL per query: 3-7 seconds

Speed Optimization Options:
"""

print("\n" + "="*70)
print("PERFORMANCE TIPS FOR FASTER RESPONSES")
print("="*70)

print("""
1. EMBEDDINGS CACHING (Already enabled)
   ✅ First run: 16s (downloads model)
   ✅ Subsequent runs: <1s (loads from cache)
   Location: ./embeddings_cache/
   
   To clear cache and rebuild:
   - Delete ./embeddings_cache/ folder
   - Next run will regenerate (16s wait)

2. FASTER LLM MODEL (Recommended)
   Currently: mistral (larger, slower)
   Alternative: gemma3 (smaller, faster)
   
   Switch in chat.py line 54:
   rag = RAGPipeline(retriever, llm_model="gemma3")
   
   Expected improvement:
   - mistral: 2-5s per query
   - gemma3: 1-2s per query
   
   To use another model:
   ollama pull llama2
   ollama pull neural-chat
   ollama pull orca-mini

3. REDUCE CONTEXT CHUNKS
   Current: k=2 (optimal for speed)
   Already optimized ✅
   
   Can reduce to k=1 for ultra-fast:
   result = rag.run(user_input, k=1)
   
4. PARALLEL PROCESSING
   Can process multiple queries
   Currently: Sequential (slower)
   Future: Could use async/threading

5. WARM UP OLLAMA
   First query is slow (model loading)
   Subsequent queries are faster
   
   Workaround:
   - Run dummy query on startup
   - Or wait 2-3 queries before heavy use

6. HARDWARE ACCELERATION
   If you have GPU:
   - Replace torch/transformers with CUDA versions
   - Update Ollama to use GPU
   - Result: 5-10x faster inference

7. QUERY OPTIMIZATION
   Shorter queries = faster responses
   "What are risks?" vs "Give detailed analysis of all risks mentioned..."
   
RECOMMENDED SETUP FOR SPEED:
=============================
1. Use gemma3 model (faster than mistral)
2. Keep k=2 chunks (already set)
3. Delete embeddings cache ONCE per session
4. Run warm-up query: "Hello"
5. Subsequent queries will be 1-2 seconds

CURRENT STATUS:
===============
""")

import os
cache_dir = "./embeddings_cache"
if os.path.exists(cache_dir):
    cache_files = os.listdir(cache_dir)
    print(f"✓ Embeddings cached ({len(cache_files)} files)")
else:
    print("⚠ Cache missing (will be created on next run)")

print("\n" + "="*70)
print("Ready to chat! Type: python chat.py")
print("="*70 + "\n")
