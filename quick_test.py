"""
Quick test of the short & best prompt
"""
import sys
sys.path.insert(0, r"c:\Users\anandsagar\OneDrive - Deloitte (O365D)\Documents\hh\cache_based_sys")

from chat import initialize_rag
from follow_up_generator import FollowUpGenerator

print("\n" + "="*70)
print("TESTING SHORT & BEST PROMPT")
print("="*70)

rag = initialize_rag()
user_q = "How many respondents neither agree nor disagree with the capability of financial advisors to handle a financial crisis?"

print(f"\nQ: {user_q}\n")

context = rag.retriever.retrieve(user_q, top_k=3)
gen = FollowUpGenerator(rag.llm)
result = gen.generate(user_q, context)

print("\nFOLLOW-UP QUESTIONS:")
print("-" * 70)
if result and 'follow_up_questions' in result:
    for i, q in enumerate(result['follow_up_questions'], 1):
        print(f"{i}. {q}")
        
    # Validate
    red_flags = ["figure", "Figure", "source", "Source", "how can", 
                 "strategies", "best practices", "according to"]
    print("\n" + "-" * 70)
    print("QUALITY CHECK:")
    issues = []
    for i, q in enumerate(result['follow_up_questions'], 1):
        for flag in red_flags:
            if flag.lower() in q.lower():
                issues.append(f"Q{i}: '{flag}' found")
    
    if issues:
        print("❌ Issues:", issues)
    else:
        print("✅ All questions are clean!")
else:
    print("❌ Generation failed")

print("="*70 + "\n")
