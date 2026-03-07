"""Conversation summarization demo using ACE Next (manual Pipeline construction).

Low-level counterpart to ``conversation_summarization_ace_next.py``.
Builds the pipeline by hand using individual steps, giving full control over
the step composition and per-epoch execution loop.

Designed for Databricks execution.

Usage (in a Databricks notebook)::

    from examples.ace_next.conversation_summarization_ace_next_pipeline import main
    main(dbutils, num_samples=20)
"""

import json
import logging
from typing import Any, Dict, List, Optional

from ace_next import (
    DeduplicationManager,
    LiteLLMClient,
    Reflector,
    Sample,
    Skillbook,
    SkillManager,
)
from ace_next.core import ACEStepContext, SkillbookView
from ace_next.steps import AgentStep, EvaluateStep, learning_tail
from pipeline import Pipeline

from conversation_summarization_ace_next import (
    SUMMARIZATION_AGENT_PROMPT,
    SummarizationAgent,
    SummarizationEnvironment,
    _make_debug_sample,
    load_summarization_tasks,
)

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = """\
You are a skillbook curator. Analyze the similar skill pairs below and decide \
how to consolidate them. Return ONLY valid JSON with a single key \
"consolidation_operations" containing a list of operations.

{report}

Current skillbook stats: {stats}
"""


MAX_SKILLS = 50


def _prune_skillbook(skillbook: Skillbook, max_skills: int = MAX_SKILLS) -> None:
    """Hard-delete lowest-scoring skills when the skillbook exceeds *max_skills*."""
    active = skillbook.skills()
    if len(active) <= max_skills:
        return
    ranked = sorted(active, key=lambda s: (s.helpful - s.harmful, s.updated_at), reverse=True)
    for skill in ranked[max_skills:]:
        skillbook.remove_skill(skill.id, soft=False)
    logger.info(
        "Pruned skillbook from %d to %d skills", len(active), max_skills
    )


def _consolidate_skills(
    dedup: DeduplicationManager,
    skillbook: Skillbook,
    llm: LiteLLMClient,
) -> None:
    """Run dedup detection, ask LLM for consolidation ops, and apply them."""
    report = dedup.get_similarity_report(skillbook)
    if not report:
        return

    prompt = CONSOLIDATION_PROMPT.format(
        report=report,
        stats=json.dumps(skillbook.stats()),
    )
    response = llm.complete(prompt)
    try:
        data: Dict[str, Any] = LiteLLMClient._extract_json(response.text)
    except (ValueError, json.JSONDecodeError):
        logger.warning("Failed to parse consolidation response")
        return

    ops = dedup.apply_operations_from_response(data, skillbook)
    if ops:
        logger.info("Consolidated %d operations, skillbook now: %s", len(ops), skillbook.stats())


def _run_pipeline(
    samples: List[Sample],
    agent_llm: LiteLLMClient,
    reasoning_llm: LiteLLMClient,
    total_epochs: int = 3,
    dedup_interval: int = 10,
) -> None:
    """Build and run the manually composed ACE pipeline on the given samples.

    Shared by :func:`main` and :func:`debug_local` — only the data loading
    and model selection differ between the two entry points.

    Args:
        agent_llm: LLM used for the summarization agent.
        reasoning_llm: LLM used for reflection, skill management, and judging.
        dedup_interval: Consolidate the skillbook every *dedup_interval* samples
            within each epoch, keeping it compact throughout training.
    """
    environment = SummarizationEnvironment(reasoning_llm)
    skillbook = Skillbook()
    dedup = DeduplicationManager()

    agent = SummarizationAgent(agent_llm)
    pipe = Pipeline(
        [
            AgentStep(agent),
            EvaluateStep(environment),
            *learning_tail(Reflector(reasoning_llm), SkillManager(reasoning_llm), skillbook),
        ]
    )

    print(f"  requires: {pipe.requires}, provides: {pipe.provides}")

    for epoch in range(1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs} ---")

        contexts = [
            ACEStepContext(
                sample=s,
                skillbook=SkillbookView(skillbook),
                epoch=epoch,
                total_epochs=total_epochs,
                step_index=0,
                total_steps=len(samples),
                global_sample_index=i,
            )
            for i, s in enumerate(samples)
        ]

        all_results = []
        for batch_start in range(0, len(contexts), dedup_interval):
            batch = contexts[batch_start : batch_start + dedup_interval]
            results = pipe.run(batch)
            pipe.wait_for_background()
            all_results.extend(results)
            _consolidate_skills(dedup, skillbook, reasoning_llm)
            _prune_skillbook(skillbook)

        errors = sum(1 for r in all_results if r.error)
        epoch_scores = environment.scores[(epoch - 1) * len(samples) : epoch * len(samples)]
        avg = sum(epoch_scores) / len(epoch_scores) if epoch_scores else 0.0

        print(f"  Processed: {len(all_results)}, Errors: {errors}, Avg score: {avg:.3f}")
        print(f"  Skills: {skillbook.stats()}")

    _consolidate_skills(dedup, skillbook, reasoning_llm)
    _prune_skillbook(skillbook)
    print(f"\nFinal skillbook: {skillbook.stats()}")
    for skill in skillbook.skills()[:5]:
        print(f"  [{skill.section}] {skill.content}")

    # Print the final prompt template with learned skills (placeholders kept for input)
    example_prompt = SUMMARIZATION_AGENT_PROMPT.format(
        skillbook=skillbook.as_prompt() or "(no strategies yet)",
        call_conversation="{call_conversation}",
        language="{language}",
        additional_instructions="{additional_instructions}",
    )
    print(f"\n--- Final prompt template ---\n{example_prompt}")

    # Run a post-learning summarization to demonstrate the agent with learned skills
    demo_sample = samples[0]
    print("\n--- Post-learning summarization (sample 0) ---")
    output = agent.generate(
        question=demo_sample.question,
        context=demo_sample.context,
        skillbook=skillbook,
    )
    print(output.final_answer)


def main(dbutils: object, num_samples: Optional[int] = None) -> None:
    """Run the manually composed ACE pipeline on conversation summarization tasks.

    Builds the pipeline explicitly as:
        AgentStep → EvaluateStep → ReflectStep → TagStep → UpdateStep → ApplyStep

    and drives the epoch loop by hand, giving full visibility into each step.

    Args:
        dbutils: Databricks ``dbutils`` object (available in Databricks notebooks).
        num_samples: Number of samples to load. Defaults to all available.
    """

    agent_llm = LiteLLMClient(model="openai/gpt-4o-mini", temperature=0.0)
    reasoning_llm = LiteLLMClient(model="openai/gpt-4.1-data-curation", temperature=0.0)
    samples = load_summarization_tasks(dbutils, num_samples=num_samples)
    _run_pipeline(samples, agent_llm, reasoning_llm, total_epochs=3)


def debug_local() -> None:
    """Run the pipeline locally on a single hardcoded sample — no Databricks required.

    Uses standard LiteLLM model names that work with the OpenAI API.
    Set the OPENAI_API_KEY environment variable before running.

    Usage::

        python conversation_summarization_ace_next_pipeline.py
    """
    agent_llm = LiteLLMClient(model="openai/gpt-4o-mini", temperature=0.0)
    reasoning_llm = LiteLLMClient(model="openai/gpt-4.1-data-curation", temperature=0.0)
    samples = _make_debug_sample()
    _run_pipeline(samples, agent_llm, reasoning_llm, total_epochs=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("ace_next").setLevel(logging.DEBUG)
    debug_local()
