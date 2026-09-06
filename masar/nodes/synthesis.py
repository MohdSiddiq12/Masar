"""Generate the final bilingual commute recommendation."""

from pydantic import BaseModel, Field

from masar.state import MasarState


class SynthesisOutput(BaseModel):
    message_en: str = Field(description="Short commute recommendation in English")
    message_ar: str = Field(description="Same recommendation in Arabic")


def _build_prompt(state: MasarState) -> str:
    route = state.get("recommended_route") or []
    mode = state.get("recommended_mode") or "drive"
    notes = state.get("context_notes") or "Conditions look normal for this time."
    return (
        f"Route: {' -> '.join(route) if route else 'no route computed'}\n"
        f"Recommended mode: {mode}\n"
        f"Context: {notes}\n\n"
        "Write a short, friendly commute message in English, then the same "
        "message in Arabic."
    )


def synthesis_node(state: MasarState, llm=None) -> dict:
    using_groq = llm is None
    if llm is None:
        import os

        from dotenv import load_dotenv
        from langchain_groq import ChatGroq

        load_dotenv()
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.3,
        ).with_structured_output(SynthesisOutput)
    from masar.api_report import measured_call

    prompt = _build_prompt(state)
    result = measured_call(
        "groq" if using_groq else "llm",
        "synthesis.invoke",
        {"prompt": prompt, "model": getattr(llm, "model_name", None)},
        lambda: llm.invoke(prompt),
    )
    if isinstance(result, dict):
        result = SynthesisOutput.model_validate(result)
    return {"message_en": result.message_en, "message_ar": result.message_ar}