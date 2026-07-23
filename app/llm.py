"""LLM client — points at Ollama's OpenAI-compatible endpoint by default.

Swap to any OpenAI-compatible provider (or a cloud model) by changing the
three LLM_* env vars only. Nothing else in the app changes.
"""
from collections.abc import Iterator

from openai import OpenAI

from .config import settings

_client = OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)

SYSTEM_PROMPT = (
    "You are a careful assistant that answers strictly from the provided "
    "context. Each context block is labelled with its source page. "
    "Cite the page(s) you used inline like [p.3]. "
    "If the answer is not contained in the context, say you don't know — "
    "do not use outside knowledge or guess."
)


CONDENSE_PROMPT = (
    "Given the conversation so far and a follow-up question, rewrite the "
    "follow-up as a standalone question that can be understood without the "
    "conversation — resolve pronouns and references (e.g. 'it', 'that clause') "
    "using the history. Return ONLY the rewritten question, nothing else. "
    "If the follow-up is already standalone, return it unchanged."
)


def condense_question(question: str, history: list[dict]) -> str:
    """Rewrite a follow-up into a self-contained question for retrieval.

    ``history`` is a list of ``{"role", "content"}`` prior turns. With no
    history the question is returned unchanged (no LLM call)."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    resp = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CONDENSE_PROMPT},
            {
                "role": "user",
                "content": f"Conversation:\n{convo}\n\nFollow-up: {question}",
            },
        ],
        temperature=0.0,
    )
    rewritten = (resp.choices[0].message.content or "").strip()
    return rewritten or question


def build_prompt(
    question: str, chunks: list[dict], history: list[dict] | None = None
) -> list[dict]:
    context = "\n\n".join(
        f"[Source page {c['page']}]\n{c['text']}" for c in chunks
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, with inline [p.N] citations."
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Prior turns help the model resolve conversational references; the final
    # user turn still carries the freshly retrieved context.
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})
    return messages


def stream_answer(
    question: str, chunks: list[dict], history: list[dict] | None = None
) -> Iterator[str]:
    stream = _client.chat.completions.create(
        model=settings.llm_model,
        messages=build_prompt(question, chunks, history),
        temperature=0.1,
        stream=True,
    )
    for event in stream:
        delta = event.choices[0].delta.content
        if delta:
            yield delta
