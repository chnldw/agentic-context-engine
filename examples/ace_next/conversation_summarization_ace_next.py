"""Conversation summarization demo using ACE Next (full ACE runner).

Adapts the agent-lightning APO conversation summarization example to use
ace_next's adaptive pipeline. Shows how ace_next can optimize prompting
strategies for call summarization via its skill-learning loop.

Evaluation is done exclusively via LLM-as-judge (no ground truth).
Designed for Databricks execution.

Usage (in a Databricks notebook)::

    from examples.ace_next.conversation_summarization_ace_next import main
    main(dbutils, num_samples=20)
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from ace_next import (
    ACE,
    Agent,
    EnvironmentResult,
    LiteLLMClient,
    Reflector,
    Sample,
    Skillbook,
    SkillManager,
    TaskEnvironment,
)
from ace_next.core import AgentOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline prompt template (V8A)
# ---------------------------------------------------------------------------

# Copied from ai-copilot-prompt-benchmark summarization_prompt.py.
# The {#if additional_instructions??}...{/if} conditional is removed —
# additional_instructions is always injected (empty string when absent).
V8A_PROMPT = r"""You are an expert call center analyst. Your task is to create a comprehensive summary of a customer support call.

BEFORE WRITING THE SUMMARY, explicitly identify:
1. The CLIENT's primary reason for calling
2. All distinct topics/issues discussed throughout the call (not just the conclusion)
3. Any emotions expressed by the CLIENT (frustration, confusion, satisfaction, urgency)
4. The AGENT's key responses, solutions offered, and actions taken
5. Any unresolved matters requiring follow-up

These identifications should be included as part of the "summarization" in an identifiable manner.

SUMMARY REQUIREMENTS:
- Cover the ENTIRE conversation chronologically, not just the resolution
- Give balanced attention to ALL topics discussed, regardless of when they appeared
- Include specific details only if they are explicitly stated in the transcript (names, dates, account numbers, amounts, reference IDs)
- Explicitly state the CLIENT's emotional state when evident (e.g., "The CLIENT expressed frustration about...")
- Document what the AGENT did to address each concern
- ONLY include information explicitly stated in the transcript - do not infer or fabricate details

NEXT ACTIONS:
- List concrete actions the AGENT must take after this call
- Each action should be specific and actionable
- If no follow-up is needed, return an empty list []

CRITICAL RULES:
- Write in {language}
- Never include information not present in the transcript
- If something is unclear in the transcript, do not guess - omit it
{additional_instructions}

OUTPUT FORMAT:
Respond ONLY with valid JSON, no markdown, no code blocks:
{{"summarization": "", "next_actions": ["", ""]}}

TRANSCRIPT:
```
{call_conversation}
```
"""


# ---------------------------------------------------------------------------
# LLM-as-a-judge grader
# ---------------------------------------------------------------------------

SUMMARIZATION_LLM_AS_A_JUDGE_PROMPT = r"""
# Task
You are provided with a **call transcript** and a **summarization** of that call.
Your goal is to **evaluate** how accurately and comprehensively the summarization reflects the content and meaning of the transcript, according to the following criteria:

1. **Full Conversation Coverage**
   - The summarization should summarize all parts of the conversation rather than focusing only on a single segment or the conclusion.

2. **Balanced Treatment of Topics**
   - The summarization should represent all topics discussed in the call fairly, without omitting or overemphasizing certain points.

3. **Essential Details**
   - The summarization should capture key issues, inquiries, and data points accurately, without oversimplification or omission.

4. **Agent Responses**
   - The summarization should adequately reflect the **AGENT**'s answers, guidance, or actions taken in response to the **CLIENT**'s concerns.

5. **Emotional Context**
   - The summarization should mention frustrations, doubts, or other emotions from the **CLIENT**, as well as how the **AGENT** addressed them.

6. **Accuracy / Hallucination**
   - The summarization should be factual and free from fabricated details or "hallucinations" not supported by the conversation.
   - **Note**: If any hallucination is detected, this should heavily lower the overall score, regardless of other strengths.

---

# Instructions

Follow these three steps:

## 1. **Analysis**
- Briefly review both the transcript and the summarization.
- Note how the summarization covers or misses each of the above criteria.
- Identify any significant omissions, distortions, or strong points related to its accuracy.

## 2. **Reasoning**
- Provide a concise explanation of how effectively the summarization meets the criteria.
- Highlight specific strengths or weaknesses.
- Mention whether there are any indications of hallucinated or fabricated content.

## 3. **Score**
- Assign a **score between 0 and 100** to reflect the summarization's overall alignment with the evaluation criteria:
  - **100** indicates a flawless summarization that addresses every criterion accurately and includes no hallucinatory content.
  - **0** indicates it completely fails to meet the criteria or is entirely fabricated.
  - If hallucinations are present, the score should be heavily reduced, regardless of other positive factors.

---

# Output Format

Your response **must strictly follow** this JSON format for structured evaluation:
{{
  "analysis": "[Brief analysis of how the summarization addresses the criteria]",
  "reasoning": "[Concise reasoning on strengths, weaknesses, and any signs of hallucination]",
  "score": <int>
}}

---

# call transcript
```
{conversation}
```

# summarization
```
{generated_summary}
```
"""


class JudgeResponse(BaseModel):
    analysis: str = Field(description="Brief analysis of how the summarization addresses the criteria.")
    reasoning: str = Field(description="Concise reasoning on strengths, weaknesses, and any signs of hallucination.")
    score: int = Field(description="The score on a 0-100 scale.")


def summarization_grader(
    judge_llm: LiteLLMClient,
    generated_summary: Optional[str],
    conversation: str,
) -> JudgeResponse:
    """Score a generated summary against the original conversation transcript.

    Uses an LLM-as-judge that evaluates coverage, balance, detail accuracy,
    agent response representation, emotional context, and hallucination.

    Returns a JudgeResponse with analysis, reasoning, and a 0-100 score.
    """
    judge_prompt = SUMMARIZATION_LLM_AS_A_JUDGE_PROMPT.format(
        conversation=conversation,
        generated_summary=generated_summary or "",
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            return judge_llm.complete_structured(judge_prompt, JudgeResponse)
        except Exception:
            logger.warning("Judge attempt %d/%d failed", attempt + 1, max_retries)

    raise ValueError(f"Judge failed after {max_retries} attempts")


# ---------------------------------------------------------------------------
# Custom environment
# ---------------------------------------------------------------------------


class SummarizationEnvironment(TaskEnvironment):
    """Evaluates agent summaries using an LLM-as-judge.

    Accumulates scores in ``self.scores`` for per-epoch reporting, since
    ``EvaluateStep`` only persists ``result.feedback`` (not ``result.metrics``)
    in the trace.
    """

    def __init__(self, judge_llm: LiteLLMClient) -> None:
        self.judge_llm = judge_llm
        self.scores: List[float] = []

    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        try:
            parsed = json.loads(agent_output.final_answer)
            generated_summary = parsed["summarization"]
        except (json.JSONDecodeError, KeyError, TypeError):
            self.scores.append(0.0)
            return EnvironmentResult(
                feedback='Invalid JSON — expected {"summarization": ..., "next_actions": [...]}',
                ground_truth=None,
                metrics={"score": 0.0},
            )

        call_conversation = sample.metadata.get("call_conversation", "")
        judge = summarization_grader(self.judge_llm, generated_summary, call_conversation)
        self.scores.append(float(judge.score))
        feedback = (
            f"LLM judge score: {judge.score}/100\n"
            f"Analysis: {judge.analysis}\n"
            f"Reasoning: {judge.reasoning}"
        )
        return EnvironmentResult(
            feedback=feedback,
            metrics={"score": judge.score},
        )


# ---------------------------------------------------------------------------
# Data loading (S3 + Spark — Databricks environment)
# ---------------------------------------------------------------------------

S3_BUCKET_PRD = "td-databricks-prd-eu-central-1-s3-aidatacuration"
S3_MOUNT_FOLDER = "/mnt/ai_data_curation/"
DATASETS_FOLDER = "datasets"


def _mount_s3(dbutils: object, s3_bucket: str, s3_mnt_folder: str) -> None:
    """Mount an S3 bucket in Databricks if not already mounted."""
    s3_source = f"s3a://{s3_bucket}"
    for m in dbutils.fs.mounts():  # type: ignore[union-attr]
        if Path(m.mountPoint).resolve() == Path(s3_mnt_folder).resolve():
            if m.source == s3_source:
                logger.debug("S3 bucket %s already mounted at %s", s3_bucket, s3_mnt_folder)
            else:
                dbutils.fs.unmount(s3_mnt_folder)  # type: ignore[union-attr]
                dbutils.fs.mount(s3_source, s3_mnt_folder)  # type: ignore[union-attr]
            return
    dbutils.fs.mount(s3_source, s3_mnt_folder)  # type: ignore[union-attr]


def load_summarization_tasks(
    dbutils: object,
    env: str = "prd",
    region: str = "eu",
    source_table: str = "observability",
    data_file: str = "20251210_000000-20251214_000000",
    num_samples: Optional[int] = 20,
    sample_seed: int = 42,
) -> List[Sample]:
    """Load summarization tasks from S3 parquet via Spark and return as ACE Samples.

    Args:
        dbutils: Databricks ``dbutils`` object for S3 mounting.
        env: Environment name (``"prd"``, ``"stg"``, ``"qa"``).
        region: Data region (``"eu"``, ``"us"``).
        source_table: Source table in the data lake.
        data_file: Root filename of the parquet + schema pair.
        num_samples: If set, randomly sample this many tasks.
        sample_seed: Random seed for reproducible sampling.

    Returns:
        List of ``Sample`` objects with the formatted V8A prompt as ``question``,
        ``ground_truth=None`` (judge-only eval), and raw fields in ``metadata``.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import coalesce, col, expr, lit, size
    from pyspark.sql.types import StructType

    s3_buckets = {
        "prd": S3_BUCKET_PRD,
        "stg": "td-databricks-stg-us-east-1-s3-aidatacuration",
        "qa": "td-databricks-qa-us-east-1-s3-aidatacuration",
    }
    s3_bucket = s3_buckets[env]
    _mount_s3(dbutils, s3_bucket, S3_MOUNT_FOLDER)

    data_dir = Path(f"/dbfs{S3_MOUNT_FOLDER}") / DATASETS_FOLDER / source_table / region / "summarization"
    spark_data_dir = Path(S3_MOUNT_FOLDER) / DATASETS_FOLDER / source_table / region / "summarization"

    schema_path = data_dir / f"{data_file}_schema.json"
    parquet_path = spark_data_dir / f"{data_file}.parquet"

    spark = SparkSession.builder.getOrCreate()
    with open(str(schema_path), "r") as f:
        schema = StructType.fromJson(json.loads(f.read()))

    data = spark.read.schema(schema).parquet(str(parquet_path))

    # Filter: conversation_turns >= 7
    data = data.withColumn(
        "conversation_turns", size(col("conversation_datapoint.data.messages"))
    ).filter(col("conversation_turns") >= 7)

    # Add y_true from feedback/silver_feedback (not used for evaluation, kept for metadata)
    columns = data.columns
    has_feedback = "feedback" in columns
    has_silver_feedback = "silver_feedback" in columns

    if has_feedback and has_silver_feedback:
        data = data.withColumn(
            "y_true", coalesce(col("feedback")[0]["value"], col("silver_feedback")[0]["value"])
        )
    elif has_feedback:
        data = data.withColumn("y_true", col("feedback")[0]["value"])
    elif has_silver_feedback:
        data = data.withColumn("y_true", col("silver_feedback")[0]["value"])
    else:
        logger.warning("No feedback or silver_feedback columns found — y_true will be null")
        data = data.withColumn("y_true", lit(None))

    # Extract summarization-specific columns
    data = data.withColumn(
        "conversation",
        expr("filter(io.inputs, x -> x.argument_id = 'call_conversation')[0].value"),
    )
    data = data.withColumn(
        "summary_language",
        expr("filter(io.inputs, x -> x.argument_id = 'language')[0].value"),
    )
    data = data.withColumn(
        "additional_instructions",
        expr("filter(io.inputs, x -> x.argument_id = 'additional_instructions')[0].value"),
    )

    # Filter: exclude Arabic
    data = data.filter(col("summary_language") != "ar-SA")

    if num_samples:
        count = data.count()
        fraction = min(1.0, num_samples / count)
        data = data.sample(withReplacement=False, fraction=fraction, seed=sample_seed).limit(num_samples)

    pdf = data.select("conversation", "summary_language", "additional_instructions", "y_true").toPandas()

    samples: List[Sample] = []
    for _, row in pdf.iterrows():
        call_conversation = row["conversation"] or ""
        language = row["summary_language"] or ""
        additional_instructions = row["additional_instructions"] or ""

        question = V8A_PROMPT.format(
            call_conversation=call_conversation,
            language=language,
            additional_instructions=additional_instructions,
        )

        samples.append(
            Sample(
                question=question,
                ground_truth=None,
                metadata={
                    "call_conversation": call_conversation,
                    "language": language,
                    "additional_instructions": additional_instructions,
                },
            )
        )

    logger.info("Loaded %d summarization samples", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(dbutils: object, num_samples: Optional[int] = None) -> None:
    """Run ACE Next adaptive pipeline on conversation summarization tasks.

    Args:
        dbutils: Databricks ``dbutils`` object (available in Databricks notebooks).
        num_samples: Number of samples to load. Defaults to all available.
    """
    logging.basicConfig(level=logging.INFO)

    llm = LiteLLMClient(model="gpt-5-mini", temperature=0.0)
    judge_llm = LiteLLMClient(model="o4-mini-data-curation", temperature=0.0)

    environment = SummarizationEnvironment(judge_llm)
    skillbook = Skillbook()

    ace = ACE.from_roles(
        agent=Agent(llm),
        reflector=Reflector(llm),
        skill_manager=SkillManager(llm),
        environment=environment,
        skillbook=skillbook,
    )

    samples = load_summarization_tasks(dbutils, num_samples=num_samples)
    results = ace.run(samples, epochs=3)

    print(f"\nTotal results: {len(results)}")

    # Per-epoch stats — environment.scores has len(samples) entries per epoch
    n = len(samples)
    for epoch in range(1, 4):
        epoch_scores = environment.scores[(epoch - 1) * n : epoch * n]
        avg = sum(epoch_scores) / len(epoch_scores) if epoch_scores else 0.0
        print(f"Epoch {epoch}: avg_score={avg:.1f}, n={len(epoch_scores)}")

    print(f"\nSkillbook: {skillbook.stats()}")
    for skill in skillbook.skills()[:5]:
        print(f"  [{skill.section}] {skill.content}")


if __name__ == "__main__":
    # This script requires a Databricks environment with dbutils available.
    print("This module is designed for Databricks execution.")
    print("Use main(dbutils) in a Databricks notebook.")
