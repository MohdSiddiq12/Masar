"""Context enrichment for low-confidence predictions."""

from masar.state import MasarState


def _get_social_signal(location: str) -> str:
    return "No social signal available yet (Reddit integration pending)."


def _get_nearby_events(location: str) -> list[str]:
    return []


def _build_prompt(state: MasarState, social_signal: str, events: list[str]) -> str:
    incidents = state.get("incidents") or []
    return (
        f"Location: {state['location']}\n"
        f"Congestion ratio: {state['congestion_ratio']:.2f} "
        "(1.0 = free flow, lower = more congested)\n"
        f"Weather: {state['weather_condition']}, raining: {state['is_raining']}\n"
        f"Reported incidents: {incidents or 'none reported'}\n"
        f"Nearby events: {events or 'none known'}\n"
        f"Social signal: {social_signal}\n\n"
        "In one or two sentences, explain the likely cause of this "
        "unusual congestion, in plain English."
    )


def context_node(state: MasarState, llm=None) -> dict:
    using_groq = llm is None
    if llm is None:
        import os

        from dotenv import load_dotenv
        from langchain_groq import ChatGroq

        load_dotenv()
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            temperature=0.2,
        )

    social_signal = _get_social_signal(state["location"])
    events = _get_nearby_events(state["location"])
    from masar.api_report import measured_call

    prompt = _build_prompt(state, social_signal, events)
    response = measured_call(
        "groq" if using_groq else "llm",
        "context.invoke",
        {"prompt": prompt, "model": getattr(llm, "model_name", None)},
        lambda: llm.invoke(prompt),
    )
    notes = response.content if hasattr(response, "content") else str(response)
    return {
        "context_notes": notes,
        "nearby_events": events,
        "social_signal": social_signal,
    }