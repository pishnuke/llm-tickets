from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

from haystack_integrations.components.generators.ollama import OllamaGenerator

LLM_URL="http://localhost:11434/api/generate"

def get_llm_pipeline(model: str, document_store, template: str):
  pipe = Pipeline()

  pipe.add_component("retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=3))
  pipe.add_component("prompt_builder", PromptBuilder(template=template))
  pipe.add_component("llm", OllamaGenerator(model=model, url=LLM_URL))

  pipe.connect("retriever", "prompt_builder.documents")
  pipe.connect("prompt_builder", "llm")
  return pipe
