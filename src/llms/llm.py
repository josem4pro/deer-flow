# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import load_yaml_config
from src.config.agents import LLMType

# Cache for LLM instances
_llm_cache: dict[LLMType, BaseChatModel] = {}


def _create_llm_use_conf(llm_type: LLMType, conf: Dict[str, Any]) -> BaseChatModel:
    if llm_type == "ollama":
        ollama_base_url = conf.get("ollama_base_url")
        ollama_model_name = conf.get("ollama_model_name")
        if not ollama_base_url:
            raise ValueError("ollama_base_url not found in configuration for LLM type 'ollama'")
        if not ollama_model_name:
            raise ValueError("ollama_model_name not found in configuration for LLM type 'ollama'")
        return ChatOllama(base_url=ollama_base_url, model=ollama_model_name)
    else:
        # Existing logic for OpenAI models (reasoning, basic, vision)
        llm_type_map_openai = {
            "reasoning": conf.get("REASONING_MODEL"),
            "basic": conf.get("BASIC_MODEL"),
            "vision": conf.get("VISION_MODEL"),
        }
        llm_params = llm_type_map_openai.get(llm_type)

        if not llm_params:
            raise ValueError(f"Configuration not found for LLM type: {llm_type} in OpenAI models mapping or conf file.")
        
        if not isinstance(llm_params, dict):
            raise ValueError(f"Invalid LLM configuration for {llm_type}: Expected a dictionary of parameters.")
        
        return ChatOpenAI(**llm_params)


def get_llm_by_type(
    llm_type: LLMType,
) -> BaseChatModel:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    if llm_type in _llm_cache:
        return _llm_cache[llm_type]

    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    llm = _create_llm_use_conf(llm_type, conf)
    _llm_cache[llm_type] = llm
    return llm


# In the future, we will use reasoning_llm and vl_llm for different purposes
# reasoning_llm = get_llm_by_type("reasoning")
# vl_llm = get_llm_by_type("vision")


if __name__ == "__main__":
    # Initialize LLMs for different purposes - now these will be cached
    basic_llm = get_llm_by_type("basic")
    print(basic_llm.invoke("Hello"))
