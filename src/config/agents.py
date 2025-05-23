# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Literal

# Define available LLM types
LLMType = Literal["ollama", "reasoning", "vision"]

# Define agent-LLM mapping
AGENT_LLM_MAP: dict[str, LLMType] = {
    "coordinator": "ollama",
    "planner": "ollama",
    "researcher": "ollama",
    "coder": "ollama",
    "reporter": "ollama",
    "podcast_script_writer": "ollama",
    "ppt_composer": "ollama",
    "prose_writer": "ollama",
}
