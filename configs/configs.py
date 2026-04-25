from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_model_paths(cfg: OmegaConf) -> None:
    """Относительные пути в models.* считаются от корня репозитория (cwd не важен)."""
    if not OmegaConf.select(cfg, "models"):
        return
    for key in ("encoder_path", "nli_path", "nli_onnx_quantized_path", "qwen_namer_path"):
        val = cfg.models.get(key)
        if not isinstance(val, str) or not val.strip():
            continue
        p = Path(val)
        if not p.is_absolute():
            cfg.models[key] = str((_REPO_ROOT / p).resolve())

# Baseline:
#   — NLI remap: medoid → nli_label = argmax_k cos(centroid, anchor_k) (clusterer)
#   — Гипотезы: вариант B (оценочные фразы ниже)
#   — Остальное: discovery / pipeline как в исходном baseline

config = OmegaConf.create({
    "models": {
        "encoder_path": "./scripts/models/models--cointegrated--rubert-tiny2/snapshots/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        "nli_path": "./models/rubert-nli/models--cointegrated--rubert-base-cased-nli-threeway/snapshots/920cbb52ef830e94461bf141ec2119979b6049e2",
        # ONNX путь для SentimentEngine (FP32/INT8). Существующий файл включает ORT автоматически.
        "nli_onnx_quantized_path": "./models/rubert-nli/nli_threeway_fp32.onnx",
        "qwen_namer_path": "./models/qwen2_5_1_5b_instruct/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
    "discovery": {
        "extractor": "ngram",
        "ngram_range": [1, 2],  # Униграммы и биграммы
        "keybert_top_k": 7,  # Сколько кандидатов на предложение до MMR
        "mmr_lambda": 0.6,  # Баланс релевантность/разнообразие
        "mmr_top_k": 5,  # Сколько кандидатов после MMR на предложение
        "min_word_length": 3,  # Минимальная длина слова-кандидата
        "cosine_threshold": 0.45,  # Порог косинуса для KeyBERT-скоринга
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "umap_metric": "cosine",
        "hdbscan_min_samples": 2,  # Фиксированный; min_cluster_size и umap_n_neighbors — адаптивные
        "anchor_similarity_threshold": 0.05,  # margin в _name_cluster (residual): best−2nd
        "anti_anchor_threshold": 0.01,  # Margin: отбросить если max_anti > max_anchor + margin
        "cluster_merge_threshold": 0.95,  # Евклидов порог мержа в UMAP R5 (residual кластеры)
        "multi_label_threshold": 0.4,   # cosine: span → anchor для NLI-пар
        "multi_label_max_aspects": 3,    # макс. якорей на один (sentence, span)
        "embedding_cache_max": 50000,  # LRU по строке для encode в KeyBERTScorer
        "dependency_filter_enabled": True,
        "dependency_filter_mode": "aspect_roles",
        "dependency_spacy_model": "ru_core_news_lg",
        "dependency_spacy_fallback_models": ["ru_core_news_md", "ru_core_news_sm"],
        "dependency_include_root_verbs": True,
        "dependency_include_root_adjs": True,
        "mdl_use_aicc_correction": False,
        "mdl_model_penalty_alpha": 1.0,
    },
    "discovery_runner": {
        "dataset_csv": "./data/dataset_final.csv",
        "results_dir": "./benchmark/discovery/results",
        "categories": [
            "physical_goods",
            "consumables",
            "hospitality",
            "services",
        ],
        "encoder_model": "ai-forever/sbert_large_nlu_ru",
        "encoder_batch_size": 8,
        "purity_threshold": 0.7,
        "top_n_phrases_per_cluster": 20,
        "hdbscan": {
            "min_cluster_size": 15,
            "min_samples": 5,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
    },
    "discovery_per_product": {
        "dataset_csv": "./data/dataset_final.csv",
        "results_dir": "./benchmark/discovery/results",
        "encoder_model": "ai-forever/sbert_large_nlu_ru",
        "encoder_batch_size": 8,
        "min_unique_phrases_to_cluster": 30,
        "top_n_phrases_per_cluster": 10,
        "purity_threshold": 0.7,
        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
        },
    },
    "sentiment": {
        # Baseline B: {aspect} = nli_label (якорь для medoid-кластеров)
        # Одна гипотеза (v4): используется только hypothesis_template_pos.
        "hypothesis_template_pos": "{aspect} — это хорошо",
        "hypothesis_template_neg": "{aspect} — это плохо",  # legacy, SentimentEngine v4 не использует
        "batch_size": 64,
        # Потоки CPU для onnxruntime (только при nli_onnx_quantized_path)
        "ort_intra_op_num_threads": 4,
        "nli_pair_cache_max": 50000,  # LRU (premise, hypothesis) → logits в SentimentEngine
        "score_epsilon": 0.0001,
        "pairing_strategy": "review_provenance",
        # Review-level NLI (one pair per review/aspect) + post-NLI relevance filter
        # legacy fallback; новые гипотезы должны использовать pairing_strategy
        "review_level": True,
        # v4: p_ent_pos=P(entailment), p_ent_neg=P(contradiction) → сумма = 1 - P(neutral)
        # Для single-hypothesis 0.6 слишком агрессивно и может обнулить пары.
        "relevance_threshold": 0.2,  # оставить пару, если p_ent_pos + p_ent_neg >= порога
        # Temperature scaling на логитах NLI
        "temperature": 0.7,
    },
    "math": {
        "prior_mean": 3.0,  # Нейтральный априор (MaxEnt)
        "prior_strength_max": 1,  # C_max для псевдо-выборки
        "time_decay_days": 365,
        "variance_penalty": 0.0,  # Убран из скора, уходит в UI-алерт
    },
    "fraud": {
        "length_sigmoid_k": 0.8,
        "length_sigmoid_x0": 4,
        "uniqueness_threshold": 0.95,  # Порог для Union-Find кластеров ботов
        "sim_noise_floor": 0.75,        # Фоновое сходство: ниже этого — штрафа нет
        "min_trust_weight": 0.01,
    },
    "ui": {
        "max_aspects_radar": 8,  # Максимум аспектов для радара и ползунков (топ по mentions)
    },
})

_resolve_model_paths(config)


def make_config_with_overrides(overrides: dict) -> OmegaConf:
    """
    Возвращает новый конфиг: baseline + overrides.
    Глобальный `config` не мутируется.
    """
    base = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    override_cfg = OmegaConf.create(overrides or {})
    merged = OmegaConf.merge(base, override_cfg)
    _resolve_model_paths(merged)
    return merged


def _replace_global_config(payload: Dict[str, Any]) -> None:
    for key in list(config.keys()):
        del config[key]
    for key, value in payload.items():
        config[key] = value


@contextmanager
def temporary_config_overrides(overrides: Dict[str, Any]) -> Iterator[OmegaConf]:
    base = OmegaConf.to_container(config, resolve=True)
    merged = make_config_with_overrides(overrides or {})
    merged_dict = OmegaConf.to_container(merged, resolve=True)
    _replace_global_config(merged_dict)
    try:
        yield merged
    finally:
        _replace_global_config(base)
