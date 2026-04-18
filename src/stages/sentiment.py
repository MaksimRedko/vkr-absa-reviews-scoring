"""
Модуль NLI Sentiment (v5 — dual hypothesis, ONNX/PyTorch + LRU cache).

Полярность через две гипотезы на пару (sentence, aspect):
  H_pos = "{aspect} — это хорошо"
  H_neg = "{aspect} — это плохо"

Для каждой гипотезы берём P(entailment) и считаем:
  Score = 1 + 4 * P_ent_pos / (P_ent_pos + P_ent_neg + eps)

Кэш LRU хранит логиты по ключу (premise, hypothesis) и работает для ONNX/PyTorch.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from configs.configs import config
from src.schemas.models import SentimentPair, SentimentResult
from src.stages.contracts import SentimentStage

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


class SentimentEngine(SentimentStage):
    """
    NLI-based sentiment engine v5 (dual hypothesis).

    Два сравниваемых утверждения для каждого аспекта:
    положительное (H_pos) и отрицательное (H_neg).
    Score = 1 + 4 * P_ent_pos / (P_ent_pos + P_ent_neg + eps)

    Инференс: PyTorch (GPU/CPU) или ONNX Runtime INT8 на CPU, если задан
    `config.models.nli_onnx_quantized_path` и файл существует.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.models.nli_path,
            local_files_only=True,
        )
        hf_cfg = AutoConfig.from_pretrained(
            config.models.nli_path,
            local_files_only=True,
        )
        self.num_labels = int(hf_cfg.num_labels)
        self.id2label = hf_cfg.id2label
        self.ent_idx, self.neu_idx, self.contra_idx = self._label_indices(
            self.id2label, self.num_labels
        )

        self.h_pos: str = config.sentiment.hypothesis_template_pos
        self.h_neg: str = config.sentiment.hypothesis_template_neg
        self.batch_size: int = config.sentiment.batch_size
        self.epsilon: float = config.sentiment.score_epsilon
        self.temperature: float = float(config.sentiment.temperature)

        self._nli_cache: OrderedDict[tuple[str, str], np.ndarray] = OrderedDict()
        self._nli_cache_max = int(
            getattr(config.sentiment, "nli_pair_cache_max", 50000) or 0
        )

        self.model: Optional[AutoModelForSequenceClassification] = None
        self._ort_session = None
        self._use_onnx = False

        onnx_path_str = getattr(config.models, "nli_onnx_quantized_path", "") or ""
        onnx_path = Path(onnx_path_str) if onnx_path_str.strip() else None

        try:
            import onnxruntime as ort  # type: ignore import-not-found
        except ImportError:
            ort = None

        if (
            onnx_path is not None
            and onnx_path.is_file()
            and ort is not None
        ):
            opts = ort.SessionOptions()
            threads = int(
                getattr(config.sentiment, "ort_intra_op_num_threads", 4) or 0
            )
            if threads > 0:
                opts.intra_op_num_threads = threads
            self._ort_session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._use_onnx = True
            onnx_kind = "INT8" if "int8" in onnx_path.name.lower() else "FP32"
            print(
                f"[SentimentEngine v5] ONNX {onnx_kind}, dual-hypothesis, {self.num_labels} classes, "
                f"ent_idx={self.ent_idx}, neu_idx={self.neu_idx}, contra_idx={self.contra_idx}, "
                f"model={onnx_path.name}"
            )
        else:
            if onnx_path_str.strip() and onnx_path is not None and not onnx_path.is_file():
                print(
                    f"[SentimentEngine v4] ONNX не найден ({onnx_path}), "
                    "используется PyTorch NLI."
                )
            elif onnx_path_str.strip() and ort is None:
                print(
                    "[SentimentEngine v4] onnxruntime не установлен — PyTorch NLI."
                )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.models.nli_path,
                local_files_only=True,
            ).to(self.device)
            self.model.eval()
            print(
                f"[SentimentEngine v5] PyTorch, dual-hypothesis, {self.num_labels} classes, "
                f"ent_idx={self.ent_idx}, neu_idx={self.neu_idx}, contra_idx={self.contra_idx}, device={self.device}"
            )

    @staticmethod
    def _label_indices(id2label: dict, num_labels: int) -> Tuple[int, int, int]:
        ent_idx = 0
        neu_idx = 1 if num_labels > 1 else 0
        contra_idx = 0
        for idx, label in id2label.items():
            lab = str(label).lower()
            if lab == "entailment":
                ent_idx = int(idx)
            if lab == "neutral":
                neu_idx = int(idx)
            if lab == "contradiction":
                contra_idx = int(idx)
        if num_labels >= 3 and len({ent_idx, neu_idx, contra_idx}) == 3:
            return ent_idx, neu_idx, contra_idx
        # fallback для стандартного порядка threeway: contradiction, neutral, entailment
        if num_labels >= 3:
            return 2, 1, 0
        return ent_idx, neu_idx, contra_idx

    def batch_analyze(
        self,
        pairs: List[SentimentPair],
    ) -> List[SentimentResult]:
        if not pairs:
            return []

        results = []
        n_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        batch_indices = range(0, len(pairs), self.batch_size)
        for i in tqdm(
            batch_indices,
            desc="      NLI батчи",
            total=n_batches,
            leave=False,
            unit="batch",
        ):
            batch = pairs[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)

        return results

    def _forward_logits_tensor(
        self, premises: List[str], hypotheses: List[str]
    ) -> torch.Tensor:
        """Сырые логиты (B, num_labels); LRU по (premise, hypothesis)."""
        assert len(premises) == len(hypotheses)
        n = len(premises)
        if n == 0:
            return torch.zeros(0, self.num_labels)

        if self._nli_cache_max <= 0:
            return self._uncached_forward_logits_tensor(premises, hypotheses)

        keys = list(zip(premises, hypotheses))
        row_vecs: List[Optional[np.ndarray]] = [None] * n
        miss_idx: List[int] = []

        for i, key in enumerate(keys):
            if key in self._nli_cache:
                self._nli_cache.move_to_end(key)
                row_vecs[i] = self._nli_cache[key]
            else:
                miss_idx.append(i)

        for start in range(0, len(miss_idx), self.batch_size):
            chunk_i = miss_idx[start : start + self.batch_size]
            sub_p = [premises[i] for i in chunk_i]
            sub_h = [hypotheses[i] for i in chunk_i]
            logits = self._uncached_forward_logits_tensor(sub_p, sub_h)
            mat = logits.detach().cpu().numpy().astype(np.float32)
            for row, orig_i in enumerate(chunk_i):
                key = keys[orig_i]
                vec = mat[row].copy()
                self._nli_cache[key] = vec
                self._nli_cache.move_to_end(key)
                while len(self._nli_cache) > self._nli_cache_max:
                    self._nli_cache.popitem(last=False)
                row_vecs[orig_i] = vec

        vecs_np = [row_vecs[i] for i in range(n)]
        assert all(v is not None for v in vecs_np)
        stacked = torch.from_numpy(np.stack(vecs_np, axis=0))
        if not self._use_onnx and self.model is not None:
            stacked = stacked.to(self.device)
        return stacked

    def _uncached_forward_logits_tensor(
        self, premises: List[str], hypotheses: List[str]
    ) -> torch.Tensor:
        """Один вызов модели / ORT без кэша."""
        if self._use_onnx:
            assert self._ort_session is not None
            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            token_type_ids = inputs.get("token_type_ids")
            if token_type_ids is None:
                token_type_ids = torch.zeros_like(inputs["input_ids"])
            ort_inputs = {
                "input_ids": inputs["input_ids"].numpy().astype(np.int64),
                "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
                "token_type_ids": token_type_ids.numpy().astype(np.int64),
            }
            logits_np = self._ort_session.run(["logits"], ort_inputs)[0]
            return torch.from_numpy(np.asarray(logits_np, dtype=np.float32))

        assert self.model is not None
        inputs = self.tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            return self.model(**inputs).logits

    def batch_collect_logits(
        self,
        pairs: List[SentimentPair],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Логиты для dual-hypothesis:
        возвращает (logits_pos, logits_neg), оба размером (N, num_labels).
        """
        if not pairs:
            empty = np.zeros((0, self.num_labels), dtype=np.float32)
            return empty, empty

        blocks_pos: List[np.ndarray] = []
        blocks_neg: List[np.ndarray] = []
        batch_indices = range(0, len(pairs), self.batch_size)
        for i in tqdm(
            batch_indices,
            desc="      NLI logits",
            total=(len(pairs) + self.batch_size - 1) // self.batch_size,
            leave=False,
            unit="batch",
        ):
            batch = pairs[i : i + self.batch_size]
            premises = [p.sentence for p in batch]
            hyp_aspects = self._hypothesis_aspects(batch)
            hyp_pos = [self.h_pos.format(aspect=a) for a in hyp_aspects]
            hyp_neg = [self.h_neg.format(aspect=a) for a in hyp_aspects]
            logits_pos = self._forward_logits_tensor(premises, hyp_pos)
            logits_neg = self._forward_logits_tensor(premises, hyp_neg)
            blocks_pos.append(logits_pos.cpu().numpy())
            blocks_neg.append(logits_neg.cpu().numpy())

        return np.vstack(blocks_pos), np.vstack(blocks_neg)

    @staticmethod
    def _hypothesis_aspects(
        batch: List[SentimentPair],
    ) -> List[str]:
        return [pair.nli_label for pair in batch]

    def _process_batch(
        self,
        batch: List[SentimentPair],
    ) -> List[SentimentResult]:
        review_ids = [p.review_id for p in batch]
        premises = [p.sentence for p in batch]
        aspects_tag = [p.aspect for p in batch]

        nli_for_hyp = self._hypothesis_aspects(batch)
        confidences = [float(p.weight) for p in batch]

        hyp_pos = [self.h_pos.format(aspect=a) for a in nli_for_hyp]
        hyp_neg = [self.h_neg.format(aspect=a) for a in nli_for_hyp]
        logits_pos = self._forward_logits_tensor(premises, hyp_pos)
        logits_neg = self._forward_logits_tensor(premises, hyp_neg)
        probs_pos = torch.softmax(logits_pos / self.temperature, dim=1).cpu().numpy()
        probs_neg = torch.softmax(logits_neg / self.temperature, dim=1).cpu().numpy()
        p_ent_pos = probs_pos[:, self.ent_idx]
        p_ent_neg = probs_neg[:, self.ent_idx]

        results = []
        for idx, (review_id, sentence, aspect_orig) in enumerate(
            zip(review_ids, premises, aspects_tag)
        ):
            pp = float(p_ent_pos[idx])
            pn = float(p_ent_neg[idx])
            score = 1.0 + 4.0 * pp / (pp + pn + self.epsilon)
            score = max(1.0, min(5.0, score))

            results.append(
                SentimentResult(
                    review_id=review_id,
                    aspect=aspect_orig,
                    sentence=sentence,
                    score=score,
                    p_ent_pos=pp,
                    p_ent_neg=pn,
                    confidence=confidences[idx],
                )
            )

        return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    sys.stdout.reconfigure(encoding="utf-8")

    from configs.configs import config  # noqa: E402

    test_pairs = [
        SentimentPair("r1", "Экран шикарный, яркие цвета", "Качество экрана", "Качество экрана"),
        SentimentPair("r2", "Сдохла батарея за день", "Батарея", "Батарея"),
        SentimentPair("r3", "Ну такое, средненько", "Общее впечатление", "Общее впечатление"),
        SentimentPair("r4", "Коробка мятая, но сам товар целый", "Логистика", "Логистика"),
        SentimentPair("r5", "Быстрая доставка, всё пришло за 2 дня", "Логистика", "Логистика"),
        SentimentPair("r6", "Качество сборки на высоте, материал отличный", "Качество", "Качество"),
    ]

    print("Инициализация SentimentEngine...")
    engine = SentimentEngine()

    print(f"\nТест на {len(test_pairs)} парах:")
    print("=" * 80)

    results = engine.batch_analyze(test_pairs)

    for res in results:
        print(f"\nReview: {res.review_id}")
        print(f"Аспект: {res.aspect}")
        print(f"Предложение: {res.sentence}")
        print(f"Score: {res.score:.2f}")
        print(f"  P(ent)={res.p_ent_pos:.3f}, P(contra)={res.p_ent_neg:.3f}")

    print("\n" + "=" * 80)
    print("Тест завершён")
