from typing import Callable

_tokenizer_registry: dict[str, type] = {}
_preprocessor_registry: dict[str, type] = {}
_converter_registry: dict[str, type] = {}


def parse_language(language: str | None) -> tuple[str, ...] | None:
    if language is None:
        return None
    return tuple(t.strip() for t in language.split(","))


def tokenizer(*, id: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if id in _tokenizer_registry:
            raise ValueError(f"Tokenizer '{id}' is already registered.")
        _tokenizer_registry[id] = cls
        return cls
    return decorator


def preprocessor(*, id: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if id in _preprocessor_registry:
            raise ValueError(f"Preprocessor '{id}' is already registered.")
        _preprocessor_registry[id] = cls
        return cls
    return decorator


def converter(*, id: str, language: str | None = None) -> Callable[[type], type]:
    tags = parse_language(language)

    def decorator(cls: type) -> type:
        if id in _converter_registry:
            raise ValueError(f"Converter '{id}' is already registered.")
        _converter_registry[id] = cls
        cls.language = tags
        return cls
    return decorator


def get_tokenizer(id: str) -> type:
    return _tokenizer_registry[id]


def get_preprocessor(id: str) -> type:
    return _preprocessor_registry[id]


def get_converter(id: str) -> type:
    return _converter_registry[id]


def list_tokenizers() -> list[str]:
    return list(_tokenizer_registry.keys())


def list_preprocessors() -> list[str]:
    return list(_preprocessor_registry.keys())


def list_converters() -> list[str]:
    return list(_converter_registry.keys())
