import json
from importlib import import_module
from typing import Optional

# provider name -> (module, class, pip package) of the LangChain integration
CHAT_PROVIDERS = {
    "openai": ("langchain_openai", "ChatOpenAI", "langchain-openai"),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI", "langchain-openai"),
    "openai_compatible": ("langchain_openai", "ChatOpenAI", "langchain-openai"),
    "ollama": ("langchain_ollama", "ChatOllama", "langchain-ollama"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic", "langchain-anthropic"),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai"),
    "mistralai": ("langchain_mistralai", "ChatMistralAI", "langchain-mistralai"),
}

EMBEDDINGS_PROVIDERS = {
    "openai": ("langchain_openai", "OpenAIEmbeddings", "langchain-openai"),
    "azure_openai": ("langchain_openai", "AzureOpenAIEmbeddings", "langchain-openai"),
    "openai_compatible": ("langchain_openai", "OpenAIEmbeddings", "langchain-openai"),
    "ollama": ("langchain_ollama", "OllamaEmbeddings", "langchain-ollama"),
    "huggingface": ("langchain_huggingface", "HuggingFaceEmbeddings", "langchain-huggingface"),
    "google_genai": ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", "langchain-google-genai"),
    "mistralai": ("langchain_mistralai", "MistralAIEmbeddings", "langchain-mistralai"),
}

# providers w/ a native JSON output mode that is enabled by default (json_mode=None)
JSON_MODE_DEFAULT_PROVIDERS = ("openai", "azure_openai", "ollama")

# config fields w/ one of these name segments are dropped (e.g. 'openai_api_key', but not 'max_tokens')
SECRET_KEY_PARTS = ("key", "token", "secret", "password", "credentials")


def _load_class(registry: dict, provider: str):
    """
    Imports the LangChain integration class registered for a provider.

    :param registry: Provider registry (CHAT_PROVIDERS or EMBEDDINGS_PROVIDERS)
    :param provider: Provider name
    :returns: LangChain model class
    :raises ValueError: Provider is unknown
    :raises ImportError: Integration package of the provider is not installed
    """

    if provider not in registry:
        raise ValueError(f"Unknown provider '{provider}', choose one of: {', '.join(registry)} "
                         "(or pass any LangChain model instance to AutoCimKG directly)")
    module_name, class_name, package = registry[provider]
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Provider '{provider}' requires the package '{package}' (pip install {package})") from e
    return getattr(module, class_name)


def create_chat_model(provider: str, model: str, temperature: float = 0, json_mode: Optional[bool] = None,
                      api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
    """
    Creates a LangChain chat model for the given provider.
    Local models are served by Ollama (provider 'ollama') or by any server offering an OpenAI-compatible API,
    such as LM Studio, vLLM, llama.cpp or LocalAI (provider 'openai_compatible' w/ base_url).

    :param provider: One of CHAT_PROVIDERS (e.g. 'openai', 'anthropic', 'ollama', 'openai_compatible')
    :param model: Model name of the provider (e.g. 'gpt-4o', 'claude-sonnet-5', 'qwen3.5:9b')
    :param temperature: Sampling temperature (defaults to 0 for reproducible extraction)
    :param json_mode: Force JSON output where the provider supports it (defaults to on for openai, azure_openai
                      and ollama; off otherwise, as the prompt itself requests JSON)
    :param api_key: API key (defaults to the provider's environment variable, e.g. OPENAI_API_KEY)
    :param base_url: Endpoint URL (required for 'openai_compatible', optional for 'ollama')
    :param kwargs: Further arguments handed to the LangChain model class
    :returns: LangChain chat model
    """

    model_class = _load_class(CHAT_PROVIDERS, provider)
    if json_mode is None:
        json_mode = provider in JSON_MODE_DEFAULT_PROVIDERS
    args = {"temperature": temperature, **kwargs}

    if provider in ("openai", "azure_openai", "openai_compatible"):
        if provider == "azure_openai":
            args["azure_deployment"] = model
        else:
            args["model"] = model
        if provider == "openai_compatible":
            if not base_url:
                raise ValueError("Provider 'openai_compatible' requires base_url (e.g. 'http://localhost:1234/v1')")
            api_key = api_key or "not-needed"  # local servers usually ignore the key, but the client demands one
        if api_key: args["api_key"] = api_key
        if base_url: args["base_url"] = base_url
        if json_mode:
            args.setdefault("model_kwargs", {})["response_format"] = {"type": "json_object"}
    elif provider == "ollama":
        args["model"] = model
        if base_url: args["base_url"] = base_url
        if json_mode: args["format"] = "json"
    else:  # anthropic, google_genai, mistralai
        args["model"] = model
        if api_key: args["api_key"] = api_key
        if base_url: args["base_url"] = base_url
        if provider == "anthropic":
            args.setdefault("max_tokens", 8192)  # default of 1024 truncates larger extraction results

    return model_class(**args)


def create_embeddings_model(provider: str, model: str, api_key: Optional[str] = None,
                            base_url: Optional[str] = None, **kwargs):
    """
    Creates a LangChain embeddings model for the given provider.
    Local embeddings are computed by Ollama (provider 'ollama'), an OpenAI-compatible server (provider
    'openai_compatible' w/ base_url) or in-process w/ sentence-transformers (provider 'huggingface').
    NOTE: all embeddings of a KG must come from the same model, including those stored w/ an existing KG!

    :param provider: One of EMBEDDINGS_PROVIDERS (e.g. 'openai', 'ollama', 'huggingface')
    :param model: Model name of the provider (e.g. 'text-embedding-3-large', 'nomic-embed-text')
    :param api_key: API key (defaults to the provider's environment variable, e.g. OPENAI_API_KEY)
    :param base_url: Endpoint URL (required for 'openai_compatible', optional for 'ollama')
    :param kwargs: Further arguments handed to the LangChain model class
    :returns: LangChain embeddings model
    """

    model_class = _load_class(EMBEDDINGS_PROVIDERS, provider)
    args = dict(kwargs)

    if provider in ("openai", "azure_openai", "openai_compatible"):
        if provider == "azure_openai":
            args["azure_deployment"] = model
        else:
            args["model"] = model
        if provider == "openai_compatible":
            if not base_url:
                raise ValueError("Provider 'openai_compatible' requires base_url (e.g. 'http://localhost:1234/v1')")
            api_key = api_key or "not-needed"
            # send raw text instead of tiktoken token ids, which only OpenAI's endpoint understands
            args.setdefault("check_embedding_ctx_length", False)
        if api_key: args["api_key"] = api_key
        if base_url: args["base_url"] = base_url
    elif provider == "ollama":
        args["model"] = model
        if base_url: args["base_url"] = base_url
    elif provider == "huggingface":
        args["model_name"] = model
    else:  # google_genai, mistralai
        args["model"] = model
        if api_key: args["google_api_key" if provider == "google_genai" else "api_key"] = api_key

    return model_class(**args)


def get_model_config(model) -> dict:
    """
    Describes a chat or embeddings model as a JSON-serialisable dict w/o secrets (e.g. API keys),
    suited for logging and for storing it as LLM config in the metadata repository.

    :param model: LangChain chat or embeddings model (or any other object)
    :returns: Model configuration w/ secrets removed
    """

    try:
        config = json.loads(model.model_dump_json())
    except Exception:  # not a (serialisable) pydantic model
        try:
            config = json.loads(json.dumps(model.model_dump(), default=str))
        except Exception:
            config = {}
    config = {key: value for key, value in config.items()
              if not any(part in key.lower().split("_") for part in SECRET_KEY_PARTS)}
    config["class"] = f"{type(model).__module__}.{type(model).__name__}"
    return config
