"""Prompt constants for Hermes Context Manager v2."""

# Injected as additional context on each turn via pre_llm_call.
# Kept minimal — the model doesn't need to know about context management.
SYSTEM_CONTEXT = (
    "Context management is handled automatically in the background. "
    "You do not need to manage context yourself."
)
