import sys, time, sqlite3
sys.stdout.reconfigure(encoding="utf-8")

from src.discovery.candidates import CandidateExtractor
from src.discovery.scorer import KeyBERTScorer, ScoredCandidate
from src.discovery.clusterer import AspectClusterer

# Тестовый набор: разные товары и объёмы
TEST_CASES = [
    # (nm_id, limit, description)
    (154532597, 200, "Гитара (мало)"),
    (154532597, 600, "Гитара (средне)"),
    (154532597, 1000, "Гитара (много)"),
    (297799336, 300, "Вибратор (мало)"),
    (297799336, 800, "Вибратор (много)"),
    (123645159, 400, "Товар 123645159 (средне)"),
    (114957337, 500, "Товар 114957337 (средне)"),
]

extractor = CandidateExtractor()
scorer = KeyBERTScorer()
clusterer = AspectClusterer(model=scorer.model)

for nm_id, limit, description in TEST_CASES:
    print(f"\n{'='*80}")
    print(f"ТОВАР: {description}")
    print(f"nm_id={nm_id}, limit={limit}")
    print('='*80)
    
    conn = sqlite3.connect("data/dataset.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT full_text FROM reviews WHERE nm_id=? AND full_text IS NOT NULL AND full_text != '' LIMIT ?",
        (nm_id, limit),
    )
    texts = [row[0] for row in cur.fetchall()]
    conn.close()
    
    print(f"Загружено отзывов: {len(texts)}")
    
    t0 = time.time()
    all_scored: list[ScoredCandidate] = []
    for text in texts:
        cands = extractor.extract(text)
        scored = scorer.score_and_select(cands)
        all_scored.extend(scored)
    
    t1 = time.time()
    print(f"Scored-кандидатов: {len(all_scored)} за {t1 - t0:.1f}с")
    
    aspects = clusterer.cluster(all_scored, min_mentions=3)
    
    t2 = time.time()
    print(f"Кластеризация: {t2 - t1:.1f}с")
    print(f"Найдено аспектов: {len(aspects)}")
    print()
    
    for name, info in sorted(aspects.items(), key=lambda x: len(x[1].keywords), reverse=True)[:10]:
        kw_sample = info.keywords[:6]
        print(f"  [{name}] ({len(info.keywords)} kw): {kw_sample}")
    
    if len(aspects) > 10:
        print(f"  ... ещё {len(aspects) - 10} аспектов")

print(f"\n{'='*80}")
print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
print('='*80)
