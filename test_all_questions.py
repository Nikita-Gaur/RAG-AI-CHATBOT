"""
Test prompt with multiple question types to verify it's universally good
"""
import sys
sys.path.insert(0, r"c:\Users\anandsagar\OneDrive - Deloitte (O365D)\Documents\hh\cache_based_sys")

from chat import initialize_rag
from follow_up_generator import FollowUpGenerator

print("\n" + "="*80)
print("COMPREHENSIVE PROMPT TEST - MULTIPLE QUESTION TYPES")
print("="*80)

rag = initialize_rag()
gen = FollowUpGenerator(rag.llm)

# Test questions of different types
test_questions = [
    "How many respondents neither agree nor disagree with the capability of financial advisors to handle a financial crisis?",
    "What demographic factors influence investment decisions?",
    "Are there differences in trust levels between age groups?",
    "What percentage of respondents believe financial advisors can handle crises?",
    "How do financial advisors influence investment decisions?"
]

red_flags = ["figure", "Figure", "source", "Source", "how can", 
             "strategies", "best practices", "according to", "external",
             "should", "recommendations"]

results_summary = {
    "total": len(test_questions),
    "clean": 0,
    "issues": []
}

for idx, question in enumerate(test_questions, 1):
    print(f"\nTest {idx}: {question[:60]}...")
    print("-" * 80)
    
    try:
        context = rag.retriever.retrieve(question, top_k=2)
        result = gen.generate(question, context)
        
        if result and 'follow_up_questions' in result:
            questions = result['follow_up_questions']
            
            # Check for issues
            has_issues = False
            for i, q in enumerate(questions, 1):
                for flag in red_flags:
                    if flag.lower() in q.lower():
                        results_summary["issues"].append({
                            "test": idx,
                            "q_num": i,
                            "flag": flag,
                            "question": q[:50]
                        })
                        has_issues = True
            
            if not has_issues:
                results_summary["clean"] += 1
                print("✅ CLEAN - All questions grounded & analytical")
                for i, q in enumerate(questions, 1):
                    print(f"   {i}. {q[:70]}...")
            else:
                print("❌ ISSUES FOUND")
                for i, q in enumerate(questions, 1):
                    for flag in red_flags:
                        if flag.lower() in q.lower():
                            print(f"   Q{i}: Contains '{flag}'")
        else:
            print("⚠️  Generation failed")
    except Exception as e:
        print(f"⚠️  Error: {str(e)[:50]}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total tests: {results_summary['total']}")
print(f"Clean: {results_summary['clean']}/{results_summary['total']}")
print(f"Success rate: {results_summary['clean']*100//results_summary['total']}%")

if results_summary["issues"]:
    print(f"\nProblems found: {len(results_summary['issues'])}")
    for issue in results_summary["issues"]:
        print(f"  - Test {issue['test']}, Q{issue['q_num']}: '{issue['flag']}' in question")
    print("\n❌ Prompt needs improvement")
else:
    print("\n✅ Prompt works well for ALL question types!")

print("="*80 + "\n")
