from omegaconf import OmegaConf


config = OmegaConf.create({
    "models": {
        "encoder_path": "./scripts/models/models--cointegrated--rubert-tiny2/snapshots/e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae",
        "nli_path": "./scripts/models/rubert-nli/models--cointegrated--rubert-base-cased-nli-twoway/snapshots/0ba52a76fdf6929b227cf0237f1bc3b2971cc8e6",
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
        "anchor_similarity_threshold": 0.05,  # Минимальный margin best-2nd для макро-якорей (антиврун)
        "anti_anchor_threshold": 0.01,  # Margin: отбросить если max_anti > max_anchor + margin
        "cluster_merge_threshold": 0.95,  # Евклидов порог мержа в UMAP R5 (меньше = ближе)
    },
    "sentiment": {
        "hypothesis_template": "Автор доволен {aspect}",
        "batch_size": 64,
        "score_epsilon": 0.0001,
    },
    "math": {
        "prior_mean": 3.0,  # Нейтральный априор (MaxEnt)
        "prior_strength_max": 3,  # C_max для псевдо-выборки
        "time_decay_days": 365,
        "variance_penalty": 0.0,  # Убран из скора, уходит в UI-алерт
    },
    "fraud": {
        "length_sigmoid_k": 0.8,
        "length_sigmoid_x0": 4,
        "uniqueness_threshold": 0.95,
        "min_trust_weight": 0.01,
    },
})