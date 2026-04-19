from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to load vocabulary YAML") from exc


@dataclass(frozen=True, slots=True)
class AspectDefinition:
    id: str
    canonical_name: str
    synonyms: list[str]
    level: str
    domains: list[str]
    hypothesis_template: str


class Vocabulary:
    def __init__(
        self,
        aspects: list[AspectDefinition],
        *,
        _by_id: dict[str, AspectDefinition],
        _by_canonical: dict[str, AspectDefinition],
    ) -> None:
        self._aspects = aspects
        self._by_id = _by_id
        self._by_canonical = _by_canonical

    @property
    def aspects(self) -> list[AspectDefinition]:
        return list(self._aspects)

    @classmethod
    def load_from_yaml(cls, path: str | Path) -> Vocabulary:
        p = Path(path)
        raw_text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError(f"Vocabulary YAML root must be a mapping, got {type(data).__name__}")

        if "language" not in data:
            raise ValueError("Vocabulary YAML root must contain 'language'")
        if data.get("language") in (None, ""):
            raise ValueError("Vocabulary YAML root 'language' must be a non-empty string")

        if "version" not in data and "yamlversion" not in data:
            raise ValueError(
                "Vocabulary YAML root must contain 'version' or 'yamlversion' (schema version)"
            )
        ver = data.get("version", data.get("yamlversion"))
        if ver in (None, ""):
            raise ValueError("Vocabulary YAML root 'version' / 'yamlversion' must be non-empty")

        aspects_raw = data.get("aspects")
        if aspects_raw is None:
            raise ValueError("Vocabulary YAML must contain 'aspects' list")
        if not isinstance(aspects_raw, list):
            raise ValueError(f"Vocabulary 'aspects' must be a list, got {type(aspects_raw).__name__}")

        aspects: list[AspectDefinition] = []
        seen_ids: list[str] = []
        for idx, item in enumerate(aspects_raw):
            if not isinstance(item, dict):
                raise ValueError(
                    f"aspects[{idx}] must be a mapping, got {type(item).__name__}"
                )
            aid = item.get("id")
            ctx = f"aspect id={aid!r}" if aid is not None else f"aspects[{idx}]"
            aspects.append(_parse_aspect(item, ctx))
            seen_ids.append(aspects[-1].id)

        id_set = set(seen_ids)
        if len(id_set) != len(seen_ids):
            dup = sorted({i for i in seen_ids if seen_ids.count(i) > 1})
            assert False, f"duplicate aspect id(s) in vocabulary: {dup}"

        by_id = {a.id: a for a in aspects}
        by_canonical = {a.canonical_name: a for a in aspects}

        return cls(aspects, _by_id=by_id, _by_canonical=by_canonical)

    def get_by_id(self, aspect_id: str) -> AspectDefinition:
        try:
            return self._by_id[aspect_id]
        except KeyError as exc:
            raise KeyError(aspect_id) from exc

    def get_by_canonical_name(self, name: str) -> AspectDefinition:
        try:
            return self._by_canonical[name]
        except KeyError as exc:
            raise KeyError(name) from exc

    def get_by_domain(self, domain: str) -> list[AspectDefinition]:
        return [a for a in self._aspects if domain in a.domains]

    def get_synonyms(self, aspect_id: str) -> list[str]:
        return list(self.get_by_id(aspect_id).synonyms)

    def get_hypothesis(
        self, aspect_id: str, variant: Literal["mention", "positive", "negative"]
    ) -> str:
        a = self.get_by_id(aspect_id)
        if variant == "mention":
            return a.hypothesis_template
        if variant == "positive":
            return f"В этом отзыве {a.canonical_name} оценивается положительно"
        return f"В этом отзыве {a.canonical_name} оценивается отрицательно"

    def all_synonym_terms(self) -> set[str]:
        out: set[str] = set()
        for a in self._aspects:
            out.update(a.synonyms)
        return out


def _parse_aspect(item: dict, ctx: str) -> AspectDefinition:
    aid = item.get("id")
    if aid is None or not isinstance(aid, str) or not aid.strip():
        raise ValueError(f"{ctx}: 'id' must be a non-empty string")

    def err(msg: str) -> ValueError:
        return ValueError(f"aspect id={aid!r}: {msg}")

    cn = item.get("canonical_name")
    if cn is None or not isinstance(cn, str) or not cn.strip():
        raise err("'canonical_name' must be a non-empty string")

    syns = item.get("synonyms")
    if not isinstance(syns, list) or not syns:
        raise err("'synonyms' must be a non-empty list of strings")
    norm_syns: list[str] = []
    for i, s in enumerate(syns):
        if not isinstance(s, str) or not s.strip():
            raise err(f"'synonyms[{i}]' must be a non-empty string")
        norm_syns.append(s)

    level = item.get("level")
    if level not in ("general", "specific"):
        raise err("'level' must be 'general' or 'specific'")

    doms = item.get("domains")
    if not isinstance(doms, list) or not doms:
        raise err("'domains' must be a non-empty list of strings")
    norm_doms: list[str] = []
    for i, d in enumerate(doms):
        if not isinstance(d, str) or not d.strip():
            raise err(f"'domains[{i}]' must be a non-empty string")
        norm_doms.append(d)

    ht = item.get("hypothesis_template")
    if ht is None or not isinstance(ht, str) or not ht.strip():
        raise err("'hypothesis_template' must be a non-empty string")

    return AspectDefinition(
        id=aid,
        canonical_name=cn,
        synonyms=norm_syns,
        level=level,
        domains=norm_doms,
        hypothesis_template=ht,
    )
