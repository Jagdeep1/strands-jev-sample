"""Shared setup for the examples.

Two providers, two roles:
  * Amazon Bedrock serves the agent's chat models (generation, tool use, structured output).
  * OpenRouter serves Jev only (decisions).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from strands.models import BedrockModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `strands_jev` importable without installing
from strands_jev import JevClient  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Bedrock inference profile IDs. Override in .env if your account uses different ones.
FAST_MODEL = os.environ.get("FAST_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")
POWERFUL_MODEL = os.environ.get("POWERFUL_MODEL", "global.anthropic.claude-sonnet-5")
BEDROCK_REGION = os.environ.get("BEDROCK_REGION")  # None -> boto3 session / AWS_REGION / Strands default

JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")


def chat_model(model_id: str, max_tokens: int = 1024) -> BedrockModel:
    """A Strands model backed by Amazon Bedrock. Uses your default AWS credentials (env, profile, or SSO)."""
    return BedrockModel(model_id=model_id, max_tokens=max_tokens, region_name=BEDROCK_REGION)


def jev_client() -> JevClient:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set (needed for Jev). Copy .env.example to .env and add your key.")
    return JevClient(key, model=JEV_MODEL)


def enable_logs() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("strands_jev").setLevel(logging.INFO)
    logging.getLogger("strands.models.routing").setLevel(logging.INFO)
