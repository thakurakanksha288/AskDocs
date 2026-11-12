"""
Gemini LLM client with structured prompt for RAG.

Prompt design
-------------
- System instruction forbids the model from using outside knowledge.
- Each context chunk is labelled with its chunk_id so the model can cite it.
- The model is told to output "I don't know" when the context is insufficient.
- Temperature is set low (0.1) to reduce hallucination.
"""

import asyncio

import google.generativeai as genai

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import logger

settings = get_settings()

genai.configure(api_key=settings.gemini_api_key)

_SYSTEM_INSTRUCTION = (
    "You are a document Q&A assistant. Answer the user's question using ONLY the "
    "context passages provided. Each passage is labelled [chunk:<id>]. Cite the "
    "chunk ids you used in your answer like this: [chunk:abc123]. "
    "If the context does not contain enough information to answer, reply with exactly: "
    '"I don\'t know based on the provided documents." '
    "Do not use any outside knowledge."
)


def _build_prompt(question: str, chunks: list[dict]) -> str:
    context_parts: list[str] = []
    for chunk in chunks:
        context_parts.append(f"[chunk:{chunk['id']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def _call_gemini_sync(prompt: str) -> str:
    model = genai.GenerativeModel(
        model_name=settings.gemini_model,
        system_instruction=_SYSTEM_INSTRUCTION,
        generation_config=genai.GenerationConfig(
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
        ),
    )
    response = model.generate_content(prompt)
    return response.text


async def generate_answer(question: str, chunks: list[dict]) -> str:
    """Generate a cited answer from the retrieved chunks."""
    prompt = _build_prompt(question, chunks)
    loop = asyncio.get_event_loop()
    try:
        answer: str = await loop.run_in_executor(None, _call_gemini_sync, prompt)
    except Exception as exc:
        raise LLMError(f"Gemini API error: {exc}") from exc
    logger.info("llm_answer_generated", question_preview=question[:60])
    return answer
