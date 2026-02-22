"""Follow-up Questions Generator - Minimized version"""
import json
from langchain_core.prompts import ChatPromptTemplate

class FollowUpGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = self._create_prompt()
    
    def _create_prompt(self):
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
{{
    "understanding": "One-sentence restatement of the user's question",
    "answer": "Concise answer using only the CONTEXT",
    "confidence": "high | medium | low",
    "follow_up_questions": [
        "Follow-up question 1?",
        "Follow-up question 2?",
        "Follow-up question 3?"
    ]
}}
"""
        return ChatPromptTemplate.from_template(prompt_text)
    
    def generate(self, user_query: str, context_docs: list) -> dict:
        context_text = self._prepare_context(context_docs)
        try:
            response = self.llm.invoke(self.prompt_template.format_prompt(user_question=user_query, context=context_text))
            data = self._parse_json(response)
            if not data:
                return self._fallback_response("JSON parsing failed")
            
            data['follow_up_questions'] = data.get('follow_up_questions', [])[:3]
            if not self._validate_response(data):
                return self._fallback_response("Invalid response format")
            
            data['follow_up_questions'] = self._filter_questions(data['follow_up_questions'], user_query, context_text)
            if len(data['follow_up_questions']) != 3:
                data['follow_up_questions'] = data['follow_up_questions'][:3]
            return data
        except Exception as e:
            return self._fallback_response(str(e))
    
    def _parse_json(self, response: str) -> dict:
        try:
            return json.loads(response.strip())
        except:
            pass
        
        try:
            json_str = response.split("```")[1]
            if "json" in json_str:
                json_str = json_str.split("json")[1]
            return json.loads(json_str.strip())
        except:
            pass
        
        try:
            start, end = response.find('{'), response.rfind('}') + 1
            return json.loads(response[start:end]) if start >= 0 and end > start else None
        except:
            return None
    
    def _prepare_context(self, context_docs: list) -> str:
        if not context_docs:
            return "No context available"
        context_text = ""
        for i, doc in enumerate(context_docs[:2], 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context_text += f"\n[Source {i}]\n{content[:700]}\n"
        return context_text.strip() if context_text.strip() else "Context not available"
    
    def _filter_questions(self, questions: list, user_query: str, context: str) -> list:
        red_flags = ["compare with", "other studies", "external", "other research", "could be used", "strategies", 
                     "recommendations", "should", "according to experts", "other organizations", "industry standards",
                     "best practices", "how would", "if implemented", "practical steps", "strategically", "could be",
                     "further explored", "how can", "address these", "build", "improve", "potential consequences"]
        good_indicators = ["what", "which", "how many", "why", "are there", "what factors", "what percentage",
                          "differences between", "segments", "groups", "in the document", "based on", "respondents", "data shows"]
        
        scored = []
        query_kw = [kw for kw in user_query.lower().split() if len(kw) > 3]
        
        for q in questions:
            if not q or not isinstance(q, str) or len(q) < 20 or q.count("?") > 1:
                continue
            q_lower = q.lower()
            if any(flag in q_lower for flag in red_flags) or not any(kw in q_lower for kw in query_kw):
                continue
            score = sum(10 for ind in good_indicators if ind in q_lower) + min(len(q), 150)
            if any(word in q_lower for word in ["respondents", "data"]):
                score += 50
            if any(word in q_lower for word in ["demographic", "segment", "group", "age", "income"]):
                score += 40
            scored.append((score, q))
        
        return [q for _, q in sorted(scored, key=lambda x: x[0], reverse=True)[:3]] or questions[:3]
    
    def _validate_response(self, data: dict) -> bool:
        return (isinstance(data, dict) and 
                all(k in data for k in ['understanding', 'answer', 'confidence', 'follow_up_questions']) and
                isinstance(data['follow_up_questions'], list) and 
                len(data['follow_up_questions']) >= 3)
    
    def _fallback_response(self, error_msg: str) -> dict:
        return {
            "understanding": "Could not process question",
            "answer": f"Unable to generate response. {error_msg}. Please try rephrasing your question.",
            "confidence": "low",
            "follow_up_questions": [
                "Can you provide more context about your question?",
                "What specific aspect would you like to explore?",
                "Are there related topics you're interested in?"
            ]
        }
