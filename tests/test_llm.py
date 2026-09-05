import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import TestCase
from unittest.mock import patch


class FakeOpenAI:
    response = None

    def __init__(self, **_kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion)
        )

    def create_completion(self, **_kwargs):
        return self.response


def load_llm_module():
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.Llama = object

    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(error=lambda *_args, **_kwargs: None)

    openai = ModuleType("openai")
    openai.OpenAI = FakeOpenAI

    with patch.dict(
        sys.modules,
        {"llama_cpp": llama_cpp, "loguru": loguru, "openai": openai},
    ):
        sys.modules.pop("llm", None)
        return importlib.import_module("llm")


class LLMGenerateTests(TestCase):
    def test_returns_plain_text_from_compatible_provider(self):
        llm = load_llm_module()
        FakeOpenAI.response = "Provider returned plain text"

        result = llm.LLM(api_key="test", model="gpt-5.6-terra").generate([])

        self.assertEqual(result, "Provider returned plain text")

    def test_returns_content_from_standard_chat_completion(self):
        llm = load_llm_module()
        FakeOpenAI.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Standard response"))]
        )

        result = llm.LLM(api_key="test", model="gpt-5.6-terra").generate([])

        self.assertEqual(result, "Standard response")
