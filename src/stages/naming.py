from __future__ import annotations

import json
import os
import ssl
import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib import error, request

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.schemas.models import AspectInfo


SYSTEM_PROMPT = """Ключевые слова из отзывов покупателей: {keywords}
Назови одним словом, о каком свойстве товара идет речь.
Ответь ОДНИМ словом на русском."""


class ClusterNamer(ABC):
    @abstractmethod
    def rename(self, aspects: Dict[str, AspectInfo]) -> Dict[str, AspectInfo]:
        ...


class MedoidNamer(ClusterNamer):
    def rename(self, aspects: Dict[str, AspectInfo]) -> Dict[str, AspectInfo]:
        out: Dict[str, AspectInfo] = {}
        for name, info in aspects.items():
            out[name] = AspectInfo(
                keywords=list(info.keywords),
                centroid_embedding=np.asarray(info.centroid_embedding).flatten(),
                keyword_weights=list(info.keyword_weights or [1.0] * len(info.keywords)),
                nli_label=name,
            )
        return out


class LLMNamer(ClusterNamer):
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_sec: int = 30,
        max_keywords: int = 10,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.timeout_sec = int(timeout_sec)
        self.max_keywords = int(max_keywords)
        self.last_name_mapping: dict[str, Optional[str]] = {}

    @staticmethod
    def _top_keywords(info: AspectInfo, limit: int) -> List[str]:
        kws = list(info.keywords or [])
        weights = list(info.keyword_weights or [1.0] * len(kws))
        pairs = sorted(zip(kws, weights), key=lambda x: float(x[1]), reverse=True)
        return [kw for kw, _ in pairs[:limit]]

    @staticmethod
    def _normalize_response(raw: str) -> str:
        text = (raw or "").strip().replace("\n", " ")
        if not text:
            return ""
        tokenized = text.split()
        if not tokenized:
            return ""
        short = " ".join(tokenized[:2]).strip(" .,:;!?\"'")
        return short

    def _build_user_prompt(self, keywords: List[str]) -> str:
        return SYSTEM_PROMPT.format(keywords=", ".join(keywords))

    def _call_llm(self, keywords: List[str]) -> str:
        if not self.api_key:
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(keywords)},
            ],
            "temperature": 0.0,
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError):
            return ""

    @staticmethod
    def _merge_infos(left: AspectInfo, right: AspectInfo, merged_name: str) -> AspectInfo:
        lw = float(sum(left.keyword_weights or [1.0] * len(left.keywords)))
        rw = float(sum(right.keyword_weights or [1.0] * len(right.keywords)))
        total = lw + rw
        if total > 0:
            centroid = (
                lw * np.asarray(left.centroid_embedding).flatten()
                + rw * np.asarray(right.centroid_embedding).flatten()
            ) / total
        else:
            centroid = np.asarray(left.centroid_embedding).flatten()

        keywords = list(dict.fromkeys(list(left.keywords) + list(right.keywords)))
        kw_map: dict[str, float] = {}
        for kw, w in zip(left.keywords, left.keyword_weights or [1.0] * len(left.keywords)):
            kw_map[kw] = kw_map.get(kw, 0.0) + float(w)
        for kw, w in zip(right.keywords, right.keyword_weights or [1.0] * len(right.keywords)):
            kw_map[kw] = kw_map.get(kw, 0.0) + float(w)
        weights = [kw_map.get(kw, 1.0) for kw in keywords]

        return AspectInfo(
            keywords=keywords,
            centroid_embedding=np.asarray(centroid).flatten(),
            keyword_weights=weights,
            nli_label=merged_name,
        )

    def rename(self, aspects: Dict[str, AspectInfo]) -> Dict[str, AspectInfo]:
        out: Dict[str, AspectInfo] = {}
        self.last_name_mapping = {}

        for medoid_name, info in aspects.items():
            keywords = self._top_keywords(info, self.max_keywords)
            response = self._normalize_response(self._call_llm(keywords))

            if not response:
                new_name = medoid_name
            else:
                new_name = response

            self.last_name_mapping[medoid_name] = new_name
            renamed = AspectInfo(
                keywords=list(info.keywords),
                centroid_embedding=np.asarray(info.centroid_embedding).flatten(),
                keyword_weights=list(info.keyword_weights or [1.0] * len(info.keywords)),
                nli_label=new_name,
            )

            if new_name in out:
                out[new_name] = self._merge_infos(out[new_name], renamed, new_name)
            else:
                out[new_name] = renamed

        return out


class GigaChatNamer(ClusterNamer):
    def __init__(
        self,
        model: str | None = None,
        auth_key: str | None = None,
        scope: str | None = None,
        auth_url: str | None = None,
        chat_url: str | None = None,
        timeout_sec: int = 30,
        max_keywords: int = 10,
        insecure_tls: bool | None = None,
    ):
        self.model = model or os.getenv("GIGACHAT_MODEL", "GigaChat-2-Pro")
        self.auth_key = auth_key or os.getenv("GIGACHAT_AUTH_KEY", "")
        self.scope = scope or os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        self.auth_url = auth_url or os.getenv("GIGACHAT_AUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
        self.chat_url = chat_url or os.getenv("GIGACHAT_CHAT_URL", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions")
        self.timeout_sec = int(timeout_sec)
        self.max_keywords = int(max_keywords)
        self.insecure_tls = (
            bool(insecure_tls)
            if insecure_tls is not None
            else os.getenv("GIGACHAT_INSECURE", "1") == "1"
        )
        self.last_name_mapping: dict[str, Optional[str]] = {}
        self._access_token: str = ""
        self._token_expires_at_ms: int = 0

    def _ssl_context(self) -> ssl.SSLContext | None:
        if self.insecure_tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None

    @staticmethod
    def _top_keywords(info: AspectInfo, limit: int) -> List[str]:
        kws = list(info.keywords or [])
        weights = list(info.keyword_weights or [1.0] * len(kws))
        pairs = sorted(zip(kws, weights), key=lambda x: float(x[1]), reverse=True)
        return [kw for kw, _ in pairs[:limit]]

    @staticmethod
    def _normalize_response(raw: str) -> str:
        text = (raw or "").strip().replace("\n", " ")
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        return " ".join(tokens[:2]).strip(" .,:;!?\"'")

    def _build_user_prompt(self, keywords: List[str]) -> str:
        return SYSTEM_PROMPT.format(keywords=", ".join(keywords))

    @staticmethod
    def _merge_infos(left: AspectInfo, right: AspectInfo, merged_name: str) -> AspectInfo:
        lw = float(sum(left.keyword_weights or [1.0] * len(left.keywords)))
        rw = float(sum(right.keyword_weights or [1.0] * len(right.keywords)))
        total = lw + rw
        if total > 0:
            centroid = (
                lw * np.asarray(left.centroid_embedding).flatten()
                + rw * np.asarray(right.centroid_embedding).flatten()
            ) / total
        else:
            centroid = np.asarray(left.centroid_embedding).flatten()

        keywords = list(dict.fromkeys(list(left.keywords) + list(right.keywords)))
        kw_map: dict[str, float] = {}
        for kw, w in zip(left.keywords, left.keyword_weights or [1.0] * len(left.keywords)):
            kw_map[kw] = kw_map.get(kw, 0.0) + float(w)
        for kw, w in zip(right.keywords, right.keyword_weights or [1.0] * len(right.keywords)):
            kw_map[kw] = kw_map.get(kw, 0.0) + float(w)
        weights = [kw_map.get(kw, 1.0) for kw in keywords]

        return AspectInfo(
            keywords=keywords,
            centroid_embedding=np.asarray(centroid).flatten(),
            keyword_weights=weights,
            nli_label=merged_name,
        )

    def _ensure_access_token(self) -> str:
        now_ms = int(time.time() * 1000)
        if self._access_token and now_ms < max(0, self._token_expires_at_ms - 60_000):
            return self._access_token
        if not self.auth_key:
            return ""

        payload = f"scope={self.scope}".encode("utf-8")
        req = request.Request(
            url=self.auth_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Basic {self.auth_key}",
                "RqUID": str(uuid.uuid4()),
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec, context=self._ssl_context()) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            self._access_token = str(data.get("access_token", "")).strip()
            self._token_expires_at_ms = int(data.get("expires_at", 0) or 0)
            return self._access_token
        except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
            self._access_token = ""
            self._token_expires_at_ms = 0
            return ""

    def _call_llm(self, keywords: List[str]) -> str:
        token = self._ensure_access_token()
        if not token:
            return ""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(keywords)},
            ],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url=self.chat_url,
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout_sec, context=self._ssl_context()) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            return str(data["choices"][0]["message"]["content"]).strip()
        except (error.URLError, TimeoutError, KeyError, IndexError, ValueError, json.JSONDecodeError):
            return ""

    def rename(self, aspects: Dict[str, AspectInfo]) -> Dict[str, AspectInfo]:
        out: Dict[str, AspectInfo] = {}
        self.last_name_mapping = {}

        for medoid_name, info in aspects.items():
            keywords = self._top_keywords(info, self.max_keywords)
            response = self._normalize_response(self._call_llm(keywords))

            if not response:
                new_name = medoid_name
            else:
                new_name = response

            self.last_name_mapping[medoid_name] = new_name
            renamed = AspectInfo(
                keywords=list(info.keywords),
                centroid_embedding=np.asarray(info.centroid_embedding).flatten(),
                keyword_weights=list(info.keyword_weights or [1.0] * len(info.keywords)),
                nli_label=new_name,
            )

            if new_name in out:
                out[new_name] = self._merge_infos(out[new_name], renamed, new_name)
            else:
                out[new_name] = renamed
        return out


class LocalLLMNamer(ClusterNamer):
    def __init__(
        self,
        model_name: str,
        canonical_aspects: List[str],
        encoder: SentenceTransformer,
        similarity_threshold: float = 0.5,
        max_keywords: int = 10,
        max_new_tokens: int = 12,
    ):
        self.model_name = model_name
        self.canonical_aspects = list(canonical_aspects)
        self.encoder = encoder
        self.similarity_threshold = float(similarity_threshold)
        self.max_keywords = int(max_keywords)
        self.max_new_tokens = int(max_new_tokens)

        self.last_name_mapping: dict[str, Optional[str]] = {}
        self.last_raw_name_mapping: dict[str, str] = {}
        self.last_normalized_name_mapping: dict[str, str] = {}

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self._model.eval()
        self._canonical_matrix = (
            self.encoder.encode(self.canonical_aspects, show_progress_bar=False)
            if self.canonical_aspects
            else np.zeros((0, 1), dtype=np.float32)
        )

    @staticmethod
    def _top_keywords(info: AspectInfo, limit: int) -> List[str]:
        kws = list(info.keywords or [])
        weights = list(info.keyword_weights or [1.0] * len(kws))
        pairs = sorted(zip(kws, weights), key=lambda x: float(x[1]), reverse=True)
        return [kw for kw, _ in pairs[:limit]]

    @staticmethod
    def _normalize_response(raw: str) -> str:
        text = (raw or "").strip().replace("\n", " ")
        if not text:
            return ""
        tokens = text.split()
        if not tokens:
            return ""
        return tokens[0].strip(" .,:;!?\"'")

    @staticmethod
    def _merge_infos(left: AspectInfo, right: AspectInfo, merged_name: str) -> AspectInfo:
        return GigaChatNamer._merge_infos(left, right, merged_name)

    def _build_prompt(self, keywords: List[str]) -> str:
        return (
            f"Ключевые слова из отзывов: {', '.join(keywords)}\n"
            "Назови одним словом, о чем эти отзывы. Ответь ОДНИМ словом."
        )

    def _generate_name(self, keywords: List[str]) -> str:
        prompt = self._build_prompt(keywords)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        return self._normalize_response(text)

    def _canonicalize(self, raw_name: str) -> str:
        if not raw_name or self._canonical_matrix.shape[0] == 0:
            return raw_name
        emb = self.encoder.encode([raw_name], show_progress_bar=False)
        sims = cosine_similarity(emb, self._canonical_matrix)[0]
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        if best_sim >= self.similarity_threshold:
            return self.canonical_aspects[idx]
        return raw_name

    def rename(self, aspects: Dict[str, AspectInfo]) -> Dict[str, AspectInfo]:
        out: Dict[str, AspectInfo] = {}
        self.last_name_mapping = {}
        self.last_raw_name_mapping = {}
        self.last_normalized_name_mapping = {}

        for medoid_name, info in aspects.items():
            keywords = self._top_keywords(info, self.max_keywords)
            raw_name = self._generate_name(keywords)
            if not raw_name:
                raw_name = medoid_name
            final_name = self._canonicalize(raw_name)
            if not final_name:
                final_name = medoid_name

            self.last_raw_name_mapping[medoid_name] = raw_name
            self.last_normalized_name_mapping[medoid_name] = final_name
            self.last_name_mapping[medoid_name] = final_name

            renamed = AspectInfo(
                keywords=list(info.keywords),
                centroid_embedding=np.asarray(info.centroid_embedding).flatten(),
                keyword_weights=list(info.keyword_weights or [1.0] * len(info.keywords)),
                nli_label=final_name,
            )
            if final_name in out:
                out[final_name] = self._merge_infos(out[final_name], renamed, final_name)
            else:
                out[final_name] = renamed
        return out
