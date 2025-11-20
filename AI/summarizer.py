# AI/summarizer.py

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# Uses OPENAI_API_KEY from environment
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def summarize_results(summary_data: Dict[str, Any], max_words: int = 250) -> str:
    """
    Use an LLM to summarize the modeling results for the report.

    Parameters
    ----------
    summary_data : dict
        A JSON-serializable dict with the key information you want summarized
        (metrics, best models, important features, etc.).
    max_words : int
        Rough cap on the length of the summary.

    Returns
    -------
    str
        A human-readable summary suitable for a short conference-style report.
    """
    summary_json = json.dumps(summary_data, indent=2)

    prompt = f"""
You are helping write the Results section of a short machine learning project report.

The project predicts country-level GDP from environmental risk-factor features as
well as clinical statistic risk-factor features
using basic ML models (regression and possibly classification).

Here is a JSON dump of the key results, metrics, and feature importances:

{summary_json}

Write a concise summary (max ~{max_words} words) that:
- Describes how well the models perform in plain language
  (reference metrics qualitatively, don't invent numbers).
- Highlights which features are most predictive and in what direction.
- Mentions any clear patterns, strengths, or weaknesses of the models.
- Uses neutral, report-style phrasing suitable for an undergraduate ML class.

Do NOT mention JSON, code, notebooks, or implementation details.
Focus only on the analysis and findings.
"""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # New python client gives a convenience property for text output
    return response.content[0].text
