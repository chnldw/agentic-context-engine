"""Conversation summarization demo using ACE Next (manual Pipeline construction).

Low-level counterpart to ``conversation_summarization_ace_next.py``.
Builds the pipeline by hand using individual steps, giving full control over
the step composition and per-epoch execution loop.

Designed for Databricks execution.

Usage (in a Databricks notebook)::

    from examples.ace_next.conversation_summarization_ace_next_pipeline import main
    main(dbutils, num_samples=20)
"""

import logging
from typing import List, Optional

from ace_next import (
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
    SummarizationAgent,
    SummarizationEnvironment,
    _make_debug_sample,
    load_summarization_tasks,
)

logger = logging.getLogger(__name__)


def _run_pipeline(
    samples: List[Sample],
    llm: LiteLLMClient,
    judge_llm: LiteLLMClient,
    total_epochs: int = 3,
) -> None:
    """Build and run the manually composed ACE pipeline on the given samples.

    Shared by :func:`main` and :func:`debug_local` — only the data loading
    and model selection differ between the two entry points.
    """
    environment = SummarizationEnvironment(judge_llm)
    skillbook = Skillbook()

    pipe = Pipeline(
        [
            AgentStep(SummarizationAgent(llm)),
            EvaluateStep(environment),
            *learning_tail(Reflector(llm), SkillManager(llm), skillbook),
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

        results = pipe.run(contexts)

        errors = sum(1 for r in results if r.error)
        epoch_scores = environment.scores[(epoch - 1) * len(samples) : epoch * len(samples)]
        avg = sum(epoch_scores) / len(epoch_scores) if epoch_scores else 0.0

        print(f"  Processed: {len(results)}, Errors: {errors}, Avg score: {avg:.3f}")
        print(f"  Skills so far: {skillbook.stats()}")

    pipe.wait_for_background()

    print(f"\nFinal skillbook: {skillbook.stats()}")
    for skill in skillbook.skills()[:5]:
        print(f"  [{skill.section}] {skill.content}")

    # Print the final prompt that would be sent to the LLM (using first sample as example)
    strategies = skillbook.as_prompt()
    example_prompt = samples[0].question
    if strategies:
        example_prompt += f"\n\nLEARNED STRATEGIES (apply these if relevant):\n{strategies}"
    print(f"\n--- Final prompt (example, sample 0) ---\n{example_prompt}")


def main(dbutils: object, num_samples: Optional[int] = None) -> None:
    """Run the manually composed ACE pipeline on conversation summarization tasks.

    Builds the pipeline explicitly as:
        AgentStep → EvaluateStep → ReflectStep → TagStep → UpdateStep → ApplyStep

    and drives the epoch loop by hand, giving full visibility into each step.

    Args:
        dbutils: Databricks ``dbutils`` object (available in Databricks notebooks).
        num_samples: Number of samples to load. Defaults to all available.
    """
    logging.basicConfig(level=logging.INFO)

    llm = LiteLLMClient(model="gpt-5-mini", temperature=0.0)
    judge_llm = LiteLLMClient(model="o4-mini-data-curation", temperature=0.0)
    samples = load_summarization_tasks(dbutils, num_samples=num_samples)
    _run_pipeline(samples, llm, judge_llm, total_epochs=3)


def debug_local() -> None:
    """Run the pipeline locally on a single hardcoded sample — no Databricks required.

    Uses standard LiteLLM model names that work with the OpenAI API.
    Set the OPENAI_API_KEY environment variable before running.

    Usage::

        python conversation_summarization_ace_next_pipeline.py
    """
    llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.0)
    judge_llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.0)
    _run_pipeline(_make_debug_sample(), llm, judge_llm, total_epochs=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    debug_local()
