"""RLM prompt templates.

Copied from dspy/predict/rlm.py with the capacity figure and the optional
small-model tip block parameterized so we can build the instruction body in
one .format() call instead of patching upstream's output post hoc.
"""

ACTION_INSTRUCTIONS_TEMPLATE = """You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt: str) -> str` - query a sub-LLM (~{capacity_chars_k}K char capacity) for semantic analysis. `prompt` MUST be a plain string; serialize dicts/lists with `json.dumps` before passing.
- `llm_query_batched(prompts: list[str]) -> list[str]` - query multiple prompts concurrently (much faster for multiple queries). Every element of `prompts` MUST be a plain string.
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output.{small_model_tips_block}"""


CONTEXT_LENGTH_TIP = (
    "IMPORTANT: You have a total context window of approximately "
    "~{tokens_k}k tokens. Be very careful about context length limits. "
    "The sub-LLMs you can query also have this same ~{tokens_k}k token "
    "limit, so you must be conservative with how much context you send "
    "in each call. For example, a viable strategy is to feed 2-3 "
    "documents per sub-LLM query. Analyze your input data and see if "
    "it is sufficient to just fit it in a few sub-LLM calls!"
)


BATCHING_TIP = (
    "IMPORTANT: Be very careful about using 'llm_query' as it incurs "
    "high runtime costs. Always batch as much information as reasonably "
    "possible into each call (aim for around ~{capacity_chars:,} characters per "
    "call). For example, if you have 1000 lines of information to "
    "process, it's much better to split into chunks of N and call "
    "'llm_query' on each chunk rather than making 1000 individual calls. "
    "Minimize the number of 'llm_query' calls by batching related "
    "information together."
)


def build_action_instructions(
    *,
    inputs: str,
    output_fields: str,
    final_output_names: str,
    max_llm_calls: int,
    sub_lm_context_chars: int,
    small_model_tips: bool = False,
) -> str:
    """Render the RLM action instructions in a single .format() call.

    Returns the fully-formatted instruction string ready to feed into
    dspy.Signature -- no post-construction patching needed.
    """
    tips_block = ""
    if small_model_tips:
        parts = []
        if sub_lm_context_chars <= 100_000:
            parts.append(CONTEXT_LENGTH_TIP.format(tokens_k=sub_lm_context_chars // 4000))
        if sub_lm_context_chars <= 200_000:
            parts.append(BATCHING_TIP.format(capacity_chars=sub_lm_context_chars))
        if parts:
            tips_block = "\n\n" + "\n\n".join(parts)

    return ACTION_INSTRUCTIONS_TEMPLATE.format(
        inputs=inputs,
        output_fields=output_fields,
        final_output_names=final_output_names,
        max_llm_calls=max_llm_calls,
        capacity_chars_k=sub_lm_context_chars // 1000,
        small_model_tips_block=tips_block,
    )
