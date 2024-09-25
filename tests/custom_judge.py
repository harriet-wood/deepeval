import json
import time
from pydantic import BaseModel

from deepeval.models import DeepEvalBaseLLM


class CustomJudge(DeepEvalBaseLLM):
    def __init__(self, name, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.name = name

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        result = json.loads(prompt)

        return schema(**result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name


class CustomJudge2(DeepEvalBaseLLM):
    def __init__(self, name, string, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.name = name
        self.string = string

    def load_model(self):
        raise NotImplementedError

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        if prompt == "Why did the chicken cross the road?":
            time.sleep(0.1)
        result = json.loads(self.string)

        return schema(**result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.name