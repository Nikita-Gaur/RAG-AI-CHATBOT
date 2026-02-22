"""
Debug test: check why system returns fallback questions
"""
import sys
sys.path.insert(0, r"c:\Users\anandsagar\OneDrive - Deloitte (O365D)\Documents\hh\cache_based_sys")

from chat import initialize_rag
from follow_up_generator import FollowUpGenerator

print("\n" + "="*80)
print("DEBUG: Testing Follow-up Generation")
print("="*80)

# Initialize
rag = initialize_rag()
user_q = 'How many respondents neither agree nor disagree with the capability of financial advisors to handle a financial crisis?'

print(f"\nUser Question:\n{user_q}\n")

# Get context
context = rag.retriever.retrieve(user_q, top_k=3)
print(f"Retrieved {len(context)} context docs\n")

# Generate
gen = FollowUpGenerator(rag.llm)
result = gen.generate(user_q, context)

print("\nRESULT:")
print("-" * 80)
if result:
    print(f"Understanding: {result.get('understanding', 'N/A')}")
    print(f"Answer: {result.get('answer', 'N/A')[:100]}...")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"\nFollow-up Questions:")
    for i, q in enumerate(result.get('follow_up_questions', []), 1):
        print(f"  {i}. {q}")
        
    # Check if these are fallback questions
    if "Can you provide more context" in str(result.get('follow_up_questions', [])):
        print("\n⚠️ FALLBACK QUESTIONS DETECTED - Generation failed internally")
    else:
        print("\n✅ Real follow-up questions generated")
else:
    print("❌ No result returned")

print("="*80 + "\n")
