from pydantic import BaseModel


class MergedSymbolGroup(BaseModel):
    name: str
    symbols: tuple[str, ...]
