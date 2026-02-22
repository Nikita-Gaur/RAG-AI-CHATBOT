"""
Simple test: check if new prompt works with mock LLM
"""
import json
from langchain_core.prompts import ChatPromptTemplate

# Test prompt
prompt_text = """USER QUESTION: {user_question}
CONTEXT: {context}

TASK:
1) Briefly restate what the user is asking.
2) Answer the question using ONLY the information in CONTEXT.
3) Propose EXACTLY 3 distinct follow-up questions the user would naturally ask next to better understand the findings.

CONSTRAINTS:
- Use only the CONTEXT; do not invent or assume facts beyond it.
- Do not reference figures/tables unless they explicitly appear in the CONTEXT.
- Do not use prescriptive wording ("how can", "should", "recommend", "best practices"); stay descriptive and analytical.
- Each follow-up must be specific (use numbers, groups, factors, or trends where possible).
- The 3 follow-ups must cover different angles (e.g., details/metrics, comparisons between groups, reasons/implications).
- Do not repeat what has already been fully answered.

Respond ONLY with valid JSON in this exact format:
{
    "understanding": "One-sentence restatement of the user's question",
    "answer": "Concise answer using only the CONTEXT",
    "confidence": "high | medium | low",
    "follow_up_questions": [
        "Follow-up question 1?",
        "Follow-up question 2?",
        "Follow-up question 3?"
    ]
}
"""

template = ChatPromptTemplate.from_template(prompt_text)

user_question = "How many respondents neither agree nor disagree?"
context = "Survey data shows: 23% agree, 15% disagree, 18% neither agree nor disagree, 44% no opinion. This was from 500 respondents aged 25-65."

# Format the prompt
formatted = template.format_prompt(user_question=user_question, context=context)

print("="*80)
print("PROMPT FORMATTED CORRECTLY")
print("="*80)
print(formatted.to_string())
print("\n✅ Prompt formatting works - no errors")
print("="*80)
