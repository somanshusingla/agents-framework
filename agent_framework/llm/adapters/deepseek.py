from __future__ import annotations

from agent_framework.llm.adapters.openai import OpenAICompatibleChatClient


class DeepSeekChatClient(OpenAICompatibleChatClient):
    provider_name = "DeepSeek"
    default_api_base = "https://api.deepseek.com"
    default_api_key_env = "DEEPSEEK_API_KEY"
