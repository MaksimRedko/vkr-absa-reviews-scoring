"""
Диагностика Recall: проверяем, на каком этапе теряются аспекты.
Этапы: CandidateExtractor → KeyBERTScorer → HDBSCAN clustering
"""
import json
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

from src.discovery.candidates import CandidateExtractor
from src.discovery.scorer import KeyBERTScorer
from src.discovery.clusterer import AspectClusterer
from src.schemas.models import ReviewInput
from sentence_transformers import SentenceTransformer
from configs.configs import config

JSON_PATH = "parser/razmetka/longest_reviews.json"

PROBE_WORDS = {
    117808756: ["упаковка", "упаковку", "упаковке", "доставка", "доставку",
                "запах", "соответствие", "фото", "логистика"],
    254445126: ["цена", "цену", "дорого", "стоимость"],
    311233470: ["комфорт", "удобно", "удобное", "удобства"],
    506358703: ["свежесть", "свежий", "срок", "годности", "цена", "цену"],
}


def main():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        all_reviews = json.load(f)

    reviews_by_nm = defaultdict(list)
    for r in all_reviews:
        reviews_by_nm[r["nm_id"]].append(r)

    encoder = SentenceTransformer(config.models.encoder_path)
    extractor = CandidateExtractor()
    scorer = KeyBERTScorer(model=encoder)

    for nm_id, probe_words in PROBE_WORDS.items():
        raw = reviews_by_nm.get(nm_id, [])
        reviews = []
        for r in raw:
            try:
                ri = ReviewInput(**r)
                if ri.clean_text:
                    reviews.append(ri)
            except Exception:
                continue

        texts = [r.clean_text for r in reviews]

        print(f"\n{'='*70}")
        print(f"nm_id={nm_id}  ({len(reviews)} отзывов)")
        print(f"Probe words: {probe_words}")
        print(f"{'='*70}")

        # Stage 1: CandidateExtractor
        all_candidates = []
        for text in texts:
            all_candidates.extend(extractor.extract(text))
        all_spans = [c.span.lower() for c in all_candidates]

        print(f"\n[Stage 1: CandidateExtractor] {len(all_candidates)} кандидатов")
        for word in probe_words:
            hits = [s for s in all_spans if word in s]
            print(f"  '{word}': {len(hits)} совпадений", end="")
            if hits:
                unique = list(set(hits))[:8]
                print(f"  → {unique}")
            else:
                print()

        # Stage 2: KeyBERTScorer
        scored = scorer.score_and_select(all_candidates)
        scored_spans = [c.span.lower() for c in scored]

        print(f"\n[Stage 2: KeyBERTScorer] {len(scored)} после MMR")
        for word in probe_words:
            hits = [s for s in scored_spans if word in s]
            print(f"  '{word}': {len(hits)} совпадений", end="")
            if hits:
                unique = list(set(hits))[:8]
                print(f"  → {unique}")
            else:
                print("  ← ПОТЕРЯНО НА ЭТАПЕ SCORING")

        # Stage 3: Clustering
        clusterer = AspectClusterer(model=encoder)
        aspects = clusterer.cluster(scored)
        print(f"\n[Stage 3: HDBSCAN] {len(aspects)} кластеров: {list(aspects.keys())}")

        for word in probe_words:
            found_in = []
            for asp_name, info in aspects.items():
                kw_lower = [k.lower() for k in info.keywords]
                if any(word in k for k in kw_lower):
                    found_in.append(asp_name)
            if found_in:
                print(f"  '{word}': в кластере(ах) {found_in}")
            else:
                # Check if in noise cluster
                noise_count = sum(1 for s in scored_spans if word in s)
                in_any = any(
                    word in k.lower()
                    for info in aspects.values()
                    for k in info.keywords
                )
                if noise_count > 0 and not in_any:
                    print(f"  '{word}': {noise_count} scored-кандидатов, "
                          f"но ВСЕ в noise (label=-1)")
                else:
                    print(f"  '{word}': не найдено в кластерах")


if __name__ == "__main__":
    main()
