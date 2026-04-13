from omegaconf import OmegaConf

# Baseline:
#   — NLI remap: medoid → nli_label = argmax_k cos(centroid, anchor_k) (clusterer)
#   — Гипотезы: вариант B (оценочные фразы ниже)
#   — Остальное: discovery / pipeline как в исходном baseline

config = OmegaConf.create({
    "models": {
        "encoder_path": "./scripts/models/models--cointegrated--rubert-tiny2/snapshots/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        "nli_path": "./models/rubert-nli/models--cointegrated--rubert-base-cased-nli-threeway/snapshots/920cbb52ef830e94461bf141ec2119979b6049e2",
        "qwen_namer_path": "./models/qwen2_5_1_5b_instruct/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    },
    "discovery": {
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
    },
    "sentiment": {
        # Baseline B: {aspect} = nli_label (якорь для medoid-кластеров)
        "hypothesis_template_pos": "{aspect} — это хорошо",
        "hypothesis_template_neg": "{aspect} — это плохо",
        "batch_size": 64,
        "score_epsilon": 0.0001,
        # Temperature scaling на логитах NLI (Guo et al., 2017); 1.0 = без изменений
        "temperature": 1.0,
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


def make_config_with_overrides(overrides: dict) -> OmegaConf:
    """
    Возвращает новый конфиг: baseline + overrides.
    Глобальный `config` не мутируется.
    """
    base = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    override_cfg = OmegaConf.create(overrides or {})
    return OmegaConf.merge(base, override_cfg)
