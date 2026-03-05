# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # Conversation Summarization — ACE Next
# MAGIC
# MAGIC Runs the adaptive ACE Next pipeline on call-center conversation transcripts.
# MAGIC Evaluation is LLM-as-judge only (no ground truth required).
# MAGIC
# MAGIC **Usage:** Run all cells top-to-bottom. `dbutils` is provided automatically by Databricks.

# COMMAND ----------
# MAGIC %sh
# MAGIC pip install ace-framework

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import logging

logging.basicConfig(level=logging.INFO)

# Silence LiteLLM verbose output
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)

# COMMAND ----------

from examples.ace_next.conversation_summarization_ace_next import main

main(dbutils, num_samples=20)
