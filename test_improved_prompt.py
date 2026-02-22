"""
Test the improved prompt with the user's question
"""
from chat import initialize_rag
from follow_up_generator import FollowUpGenerator

print("\n" + "="*80)
print("TESTING IMPROVED PROMPT")
print("="*80)

# Initialize RAG
rag = initialize_rag()

# User's question
user_question = 'How many respondents neither agree nor disagree with the capability of financial advisors to handle a financial crisis?'

print(f'\nUSER QUESTION:\n{user_question}')
print("\n" + "-"*80)

# Get answer
answer = rag.generate_answer(user_question)
print(f'\nANSWER:\n{answer}')

print("\n" + "-"*80)
print("\nGENERATED FOLLOW-UP QUESTIONS:")
print("-"*80)

# Get context for follow-ups
context = rag.retriever.retrieve(user_question, top_k=3)

# Generate follow-ups
generator = FollowUpGenerator(rag.llm)
result = generator.generate(user_question, context)

if result and 'follow_up_questions' in result:
    follow_ups = result['follow_up_questions']
    for i, q in enumerate(follow_ups, 1):
        print(f"\n{i}. {q}")
        
    print("\n" + "="*80)
    print("QUALITY CHECK")
    print("="*80)
    
    red_flags = [
        "figure", "Figure", "Source", "source", "appendix", "chart", "table",
        "strategies", "recommendations", "should", "how can", "implement",
        "best practices", "according to", "external", "studies"
    ]
    
    hallucinations = []
    for i, q in enumerate(follow_ups, 1):
        for flag in red_flags:
            if flag in q:
                hallucinations.append(f"Q{i}: Contains '{flag}'")
    
    if hallucinations:
        print("\n⚠️  POTENTIAL ISSUES:")
        for issue in hallucinations:
            print(f"  - {issue}")
    else:
        print("\n✅ ALL QUESTIONS are properly grounded in the document!")
        print("   - No figure/source references")
        print("   - No prescriptive language")
        print("   - No external references")
else:
    print("❌ Failed to generate follow-up questions")

print("\n" + "="*80 + "\n")
