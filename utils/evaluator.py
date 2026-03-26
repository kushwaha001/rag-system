import httpx
import json
import os
from dotenv import load_dotenv

load_dotenv()

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8"))

async def evaluate_response(
    question: str,
    answer: str,
    context: str
) -> dict:
    """
    LLM-as-judge: verifies answer before finalizing.
    Returns PASS or FAIL with scores.
    """
    eval_prompt = f"""You are a strict evaluator for a RAG system. 
Evaluate if the answer is correct and grounded in the context.

Question: {question}

Context:
{context}

Answer:
{answer}

Respond ONLY with valid JSON, no extra text:
{{
    "faithfulness": <0.0-1.0, is the answer grounded in context?>,
    "hallucination": <0.0-1.0, 0=no hallucination, 1=fully hallucinated>,
    "completeness": <0.0-1.0, does it fully answer the question?>,
    "verdict": "<PASS or FAIL>",
    "reason": "<one sentence why>"
}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{VLLM_HOST}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": eval_prompt}],
                "max_tokens": 256,
                "temperature": 0.0
            }
        )
        result = response.json()
        raw = result["choices"][0]["message"]["content"].strip()

    # Parse JSON
    try:
        if "```" in raw:
            raw = raw.split("```")[1].replace("json", "").strip()
        evaluation = json.loads(raw)
    except Exception as e:
        evaluation = {
            "faithfulness": 0.0,
            "hallucination": 1.0,
            "completeness": 0.0,
            "verdict": "FAIL",
            "reason": f"Evaluator could not parse response: {str(e)}"
        }

    return evaluation
