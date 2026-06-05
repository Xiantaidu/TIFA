import json
import pathlib
from types import MappingProxyType
from typing import Iterable, Mapping, TypeVar

from lib.config.schema import MergedSymbolGroupConfig

# PAD = 0 (implicitly defined)
MASK_TOKEN = 1
SPACE_TOKEN = 2

NUM_RESERVED_TOKENS = max(MASK_TOKEN, SPACE_TOKEN) + 1

__all__ = [
    "MASK_TOKEN",
    "SPACE_TOKEN",
    "NUM_RESERVED_TOKENS",
    "VocabularyBuilder",
    "Vocabulary",
]


class VocabularyBuilder:
    def __init__(
            self,
            *,
            global_symbols: Iterable[str] = (),
            stop_symbols: Iterable[str] = (),
            merged_groups: Iterable[MergedSymbolGroupConfig] | None = None,
    ):
        self.global_symbols = frozenset(global_symbols)
        self.stop_symbols = frozenset(stop_symbols)
        self.merged_groups = list(merged_groups or ())
        self._symbol_counts: dict[str, int] = {}

    def add(self, symbols: Iterable[str], default_language: str) -> None:
        for s in symbols:
            if s in self.stop_symbols:
                continue
            if s not in self.global_symbols and "/" not in s and default_language is not None:
                s = f"{default_language}/{s}"
            if s in self.stop_symbols:
                continue
            self._symbol_counts[s] = self._symbol_counts.get(s, 0) + 1

    def counter(self) -> Mapping[str, int]:
        return MappingProxyType(self._symbol_counts)

    def build(self) -> "Vocabulary":
        observed = set(self._symbol_counts.keys())
        seen_group_names: set[str] = set()
        for group in self.merged_groups:
            if not group.name:
                raise ValueError("Merged group name cannot be empty.")
            if group.name in seen_group_names:
                raise ValueError(f"Duplicate merged group name: '{group.name}'.")
            seen_group_names.add(group.name)
            members = [str(s) for s in group.symbols]
            for s in members:
                if s in self.stop_symbols:
                    raise ValueError(
                        f"Stop symbol '{s}' cannot be a member of merged group '{group.name}'.")
            if not any(s in observed for s in members):
                raise ValueError(
                    f"None of the symbols in merged group '{group.name}' "
                    f"appear in the dataset: [{', '.join(members)}]")

        # Collect all non-stop symbols including merged group members
        all_symbols = observed.copy()
        for group in self.merged_groups:
            for s in group.symbols:
                if s not in self.stop_symbols:
                    all_symbols.add(str(s))

        # Merged-group disjoint-set
        group_map = _disjoint_sets(
            all_symbols,
            [(g.name, (str(s) for s in g.symbols if str(s) not in self.stop_symbols))
             for g in self.merged_groups],
        )

        # Assign IDs ordered by group name (plain groups use the symbol as name)
        for s in all_symbols:
            found = False
            for members in group_map.values():
                if s in members:
                    found = True
                    break
            if not found:
                group_map[s] = (s,)

        ordered_names = sorted(group_map)
        name_to_id = {name: idx for idx, name in enumerate(ordered_names, start=NUM_RESERVED_TOKENS)}

        # Build symbol -> ID mapping
        symbol_to_id: dict[str, int] = {}
        for name in ordered_names:
            gid = name_to_id[name]
            for s in group_map[name]:
                symbol_to_id[s] = gid

        return Vocabulary(
            symbol_to_id=symbol_to_id,
        )


class Vocabulary:
    def __init__(
            self,
            *,
            symbol_to_id: dict[str, int],
    ):
        self._symbol_to_id = symbol_to_id
        id_to_symbols: dict[int, list[str]] = {}
        for sym, idx in symbol_to_id.items():
            id_to_symbols.setdefault(idx, []).append(sym)
        self._id_to_symbols: dict[int, tuple[str, ...]] = {
            idx: tuple(syms) for idx, syms in id_to_symbols.items()
        }

    @property
    def vocab_size(self) -> int:
        ids = self._symbol_to_id.values()
        return max(NUM_RESERVED_TOKENS, *ids) + 1 if ids else NUM_RESERVED_TOKENS + 1

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, symbol: str, language: str | None) -> int | None:
        if symbol in self._symbol_to_id:
            return self._symbol_to_id[symbol]
        if language is not None and (prefixed_symbol := f"{language}/{symbol}") in self._symbol_to_id:
            return self._symbol_to_id[prefixed_symbol]
        return None

    def decode(self, token: int, stringfy: bool = False) -> "tuple[str, ...] | str":
        """Return the symbol(s) mapped to *token*.

        Returns a tuple of all symbols that share this ID (more than one
        when the ID represents a merged group).  If *stringfy* is True,
        joins them with ``, `` and returns a single string.
        Returns ``None`` for unknown tokens.
        """
        syms = self._id_to_symbols.get(token)
        if syms is None:
            return None
        if stringfy:
            return ", ".join(syms)
        return syms

    def to_dict(self) -> dict:
        return {
            "symbols": dict(self._symbol_to_id),
        }

    def dump(self, path: str | pathlib.Path) -> None:
        with open(path, "w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "Vocabulary":
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return cls(symbol_to_id=data["symbols"])


_T = TypeVar("_T")


def _disjoint_sets(
        elements: Iterable[_T],
        unions: Iterable[tuple[str, Iterable[_T]]],
) -> dict[str, tuple[_T, ...]]:
    """Partition *elements* according to *unions*.

    Each item in *unions* is ``(name, members)``.  Members of the same
    union are merged; unions that share members are transitively merged.
    All *names* that land in the same set map to the same sorted tuple.

    Elements not referenced by any union form a singleton set with no
    name in the result (they still participate in union-find, so they can
    be pulled into a named set by a union that references them).
    """
    parent: dict[_T, _T] = {}

    def find(x: _T) -> _T:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: _T, y: _T) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e in elements:
        parent[e] = e

    name_roots: dict[str, _T] = {}
    for name, members in unions:
        members = list(members)
        if not members:
            continue
        first = members[0]
        if first not in parent:
            parent[first] = first
        name_roots[name] = first
        for m in members[1:]:
            if m not in parent:
                parent[m] = m
            union(first, m)

    # Collect names by root
    root_to_names: dict[_T, list[str]] = {}
    for name, root in name_roots.items():
        r = find(root)
        root_to_names.setdefault(r, []).append(name)

    # Collect members by root
    root_to_members: dict[_T, list[_T]] = {}
    for e in parent:
        r = find(e)
        root_to_members.setdefault(r, []).append(e)

    result: dict[str, tuple[_T, ...]] = {}
    for root, members in root_to_members.items():
        sorted_members = tuple(sorted(members))
        for name in root_to_names.get(root, []):
            result[name] = sorted_members
    return result
