\# Compact context



\## Project

ABSA diploma. CPU-only. No LLM in inference. Path gamma.



\## Fixed decisions

\- Core + domain vocabulary accepted

\- Segmentation-first old attempt rejected

\- Current bottleneck: matching

\- New clause\_segmenter\_v1.0 accepted as baseline research artifact



\## Phase 1 result

\- physical\_goods: +3.2 p.p.

\- hospitality: +8.55 p.p.

\- services: +6.03 p.p.

\- consumables: +5.88 p.p.

\- Conclusion: weaker baseline -> stronger domain vocab gain



\## Phase 2 result

\- lexical baseline: hybrid > core-only

\- old segmentation experiment invalid, then fixed

\- candidates vs old segments: parity

\- reason: segmenter was near-noop



\## Current state

\- RuleBasedClauseSegmenter v1.0 implemented

\- Next experiment: segment surface probe (A / B1 / B2)



\## Next step

Run B1/B2 on hospitality first.

- С†РµР»СЊ СЌС‚Р°РїР°: phase2_step2b_segment_surface_matching
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: A=current candidates; B1=segment+extractor; B2=segment surface unigrams/2-3grams
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: A P/R/F1=0.4806/0.4130/0.4251; B1=РёРґРµРЅС‚РёС‡РЅРѕ A; B2=0.3758/0.3757/0.3556
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: segmentation РЅРµ РґР°Р»Р° РїСЂРёСЂРѕСЃС‚Р°; surface expansion СѓС…СѓРґС€РёР» precision Рё F1
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: A_vs_B1 diff share=0.0000; A_vs_B2=0.6311; avg preds/review A=1.0420, B2=1.6447
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: РѕСЃС‚Р°РІРёС‚СЊ A reference point; РёРґС‚Рё РІ СЃР»РµРґСѓСЋС‰РёР№ РёР·РѕР»РёСЂРѕРІР°РЅРЅС‹Р№ matching experiment
- С†РµР»СЊ СЌС‚Р°РїР°: phase2_step3 A/B lexical-only vs relaxed lexical
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: С‚РѕР»СЊРєРѕ matching; unit=candidates; hybrid vocab С„РёРєСЃРёСЂРѕРІР°РЅ
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: A P/R/F1=0.4806/0.4130/0.4251; B=0.4034/0.4469/0.4019
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: relaxed lexical РїРѕРґРЅСЏР» recall (+0.0339), РЅРѕ РїСЂРѕСЃР°РґРёР» precision Рё F1
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: baseline РѕСЃС‚Р°С‘С‚СЃСЏ A (lexical-only); diff share A/B=0.4875; avg preds/review B=1.7866
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: СЃР»РµРґСѓСЋС‰РёР№ РёР·РѕР»РёСЂРѕРІР°РЅРЅС‹Р№ matching A/B СЃ Р±РѕР»РµРµ СЃС‚СЂРѕРіРёРј relaxed-РєСЂРёС‚РµСЂРёРµРј
- С†РµР»СЊ СЌС‚Р°РїР°: phase2_step4 cosine filter РїРѕРІРµСЂС… lexical
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: A=lexical_only; B=lexical+cosine(candidate,aspect_anchor)>=tau
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: tau=0.35 no-op; tau=0.90 РґР°Р»Рѕ A 0.4806/0.4130/0.4251 vs B 0.4818/0.4125/0.4255
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: Р·Р°РјРµС‚РЅРѕРіРѕ СѓР»СѓС‡С€РµРЅРёСЏ РЅРµС‚, СЌС„С„РµРєС‚ РѕС‡РµРЅСЊ РјР°Р»
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: С„РёР»СЊС‚СЂ РЅРµ Р»РѕРјР°РµС‚ recall РїСЂРё СЃС‚СЂРѕРіРѕРј tau; baseline A РѕСЃС‚Р°С‘С‚СЃСЏ СѓСЃС‚РѕР№С‡РёРІС‹Рј reference
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: phase2_step5 (single-signal vs ensemble) РёР»Рё С‚РѕС‡РµС‡РЅС‹Р№ tau calibration РєР°Рє РѕС‚РґРµР»СЊРЅС‹Р№ РјРёРЅРё-СЌРєСЃРїРµСЂРёРјРµРЅС‚
-  : phase2_step5_cosine_margin_gate
-  : A=lexical_only; B=lexical+cosine margin gate (top1>=tau_s, top1-top2>=tau_m)
-  : A P/R/F1=0.4806/0.4130/0.4251; B=0.3968/0.3053/0.3299
-   : gate   ,   precision,  recall
-  : tau_s=0.90, tau_m=0.03; dP=-0.0838, dR=-0.1077, dF1=-0.0952
-  : baseline  lexical_only;  phase2_step5     matching-only 
-  : phase2_step5 logic fix (lexical vs nonlexical margin)
-  :  global top1/top2  best_lexical  best_nonlexical,    phase2_step5
-  :   ; A=0.4806/0.4130/0.4251, B=0.3968/0.3053/0.3299
-   :        tau_s=0.90, tau_m=0.03
-  : old B vs new B prediction diff = 0/1762
-  : baseline  lexical_only;   matching-   gate-/
-  :     phase2_step5
-  :  best_lexical, best_nonlexical, margin  TP/filtered/FP + 10 
-  : TP=671, FP=457, filtered=719; filtered    margin (669/719)
-   : TP  FP  cosine-space    score/margin
-  :  separation accepted vs filtered,   usable separation TP vs FP
-  : matching-only cosine-margin     ; baseline  lexical_only
-  :   matching-only 
-  : step3 relaxed_lexical, step4 cosine_filter, step5 cosine_margin_gate + TP/FP separation
-  :  baseline  lexical_only (F1=0.4251)
-   : relaxed_lexical  F1; cosine_filter ; margin-gate   TP  FP
-  : TP margin median 0.0419; FP margin median 0.0523;    
-  : 1) controlled multi-signal matching, 2) sentiment/aggregation    baseline
- С†РµР»СЊ СЌС‚Р°РїР°: phase2_step6 lexical_only + NLI verifier
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: A=lexical_only; B=lexical proposals + NLI verifier С‚РѕР»СЊРєРѕ РЅР° lexical hits, premise=candidate.sentence, threshold С‡РµСЂРµР· LOPO-CV
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: baseline РІРѕСЃРїСЂРѕРёР·РІС‘Р»СЃСЏ С‚РѕС‡РЅРѕ; A=0.4806/0.4130/0.4251; B=0.4806/0.4130/0.4251
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: verifier РґР°Р» РЅСѓР»РµРІРѕР№ СЌС„С„РµРєС‚; РЅРё РѕРґРёРЅ lexical Р°СЃРїРµРєС‚ РЅРµ Р±С‹Р» СЂРµР°Р»СЊРЅРѕ РѕС‚С„РёР»СЊС‚СЂРѕРІР°РЅ
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: LOPO threshold median=0.0001; 15/16 folds РІС‹Р±СЂР°Р»Рё 6.39e-05; latency 187.7 sec; NLI calls=2215
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: lexical_only С„РёРєСЃРёСЂСѓРµС‚СЃСЏ РєР°Рє final detection baseline; detection РґР°Р»СЊС€Рµ РЅРµ РєРѕРІС‹СЂСЏРµРј
- С†РµР»СЊ СЌС‚Р°РїР°: phase3_freeze_detection_and_run_sentiment_baseline
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: frozen lexical_only detection + current review-level NLI sentiment + current shrinkage aggregation
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: detection reproduced exactly 0.4806/0.4130/0.4251; review MAE=0.7116; product MAE all=0.8920; n>=3=0.7528
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: downstream coverage low; matched sentiment only for 935/4464 gold review-aspect pairs (20.95%); product MAE still above target
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: sentiment pairs=1836; NLI calls=3672; total latency=1184.15s; strong polarity inversions on negation/price/no-smell cases
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ СЌС‚РѕС‚ run РєР°Рє final frozen downstream baseline РґР»СЏ СЂРµС€РµРЅРёСЏ, С…РІР°С‚Р°РµС‚ Р»Рё quality РёР»Рё РЅСѓР¶РµРЅ РѕС‚РґРµР»СЊРЅС‹Р№ sentiment-focused СЌС‚Р°Рї
-  : phase3_step8_repair_current_segmenter
-  :   `src/stages/segmentation.py`; rule fixes   
-  : markers ` /  `, label split, ellipsis guard, `././.`, relaxed contrast guard, tail merge
-  :  failure-focused diagnostic slice 40  (10x4 )   
-  : changed outputs = 27/40; manual decision = PASS
-   :     ellipsis-heavy      
-  : repaired rule-based segmenter    stable baseline unit-of-analysis
-  : `.opencode/artifacts/phase3_step8_repair_current_segmenter/20260422_235342/`
-  :   segmenter    ;     frozen current segmenter
- goal: phase3_step9_segment_to_aspect_semantic_matching
- checked: frozen baseline vs repaired segmenter + segment semantic matching
- baseline gate: reproduced exactly 0.4806/0.4130/0.4251
- semantic setup: aspect centroid = mean(canonical+synonyms); segment = full segment embedding; score = best softmax over category aspects; threshold via LOPO
- result: semantic P/R/F1 = 0.1965/0.1748/0.1766
- delta vs baseline: -0.2841 / -0.2382 / -0.2485
- avg segments/review: 3.2372
- avg predicted aspects/review: 0.6481
- did not work: all categories and both sources degraded; hospitality collapsed most
- fixed: branch result = FAIL; lexical baseline stays active
- next step: do not replace frozen detection with segment semantic matching
- goal: phase4_anchor_residual_routing_diagnostic
- checked: old candidates + two-level anchors only; no segments, no sentiment, no aggregation
- hypothesis: general/domain routing will absorb clear candidates and leave a usable residual for later HDBSCAN
- outputs planned: routing_rules.md, candidate_routing.csv, anchor_overlap.csv, residual_pool_sample.csv, summary.md
- metric: route shares + overlap rate + residual usefulness sample
- next step: implement isolated diagnostic runner and keep baseline pipeline unchanged
- goal: phase4_anchor_residual_routing_diagnostic
- checked: old candidates -> general anchors -> domain anchors -> residual -> cleanup -> HDBSCAN eligibility
- got: thresholds fixed at Tg=0.88, Mg=0.04, Td=0.88, Md=0.04, Tgc=Tdc=0.83, Coverlap=0.02
- got: route counts general=900, domain=764, overlap=5013, residual=17394, noise=786
- did not work: overlap stayed high at 20.2%; two layers still conflict strongly
- got: residual sample 62 useful / 56 unclear / 2 noise; not garbage, but not clean post-anchor remainder
- fixed: recommendation = do_not_go_hdbscan
- next step: if returning to this branch, first reduce general-domain overlap before trusting residual clustering
- goal: phase4_step2_reduce_general_domain_overlap
- checked: next step is not HDBSCAN itself; first need to shrink general-domain overlap
- hypothesis: a small set of conflict pairs dominates overlap; targeted cleanup + domain_priority may be enough
- files planned: phase4 runner + selected vocabulary yaml files
- metric: overlap/residual before-after and residual usefulness sample
- next step: label top conflict pairs, apply only narrow vocabulary edits, rerun current vs domain_priority
- goal: phase4_step2_reduce_general_domain_overlap
- checked: top-30 overlap pairs from old diagnostic, then cleaned vocab + added domain_priority conflict mode
- got: cleanup alone moved overlap 20.2% -> 19.3%
- got: cleaned domain_priority moved overlap 20.2% -> 12.5%, domain share 3.1% -> 9.5%
- got: residual sample improved from 62/56/2 to 66/54/0 useful/unclear/noise
- fixed: recommendation switched to go_hdbscan
- saved: overlap_top30.csv, overlap_conflict_labels.csv, dictionary_changes.md, rerun_summary.md, final_recommendation.md
- next step: run residual-only HDBSCAN as the next isolated experiment
- goal: phase4_step3_residual_only_hdbscan
- checked: next isolated step is residual-only HDBSCAN after cleaned `domain_priority` routing
- hypothesis: cleaned residual can yield >=5 useful top-20 clusters without duplicate/noise majority
- scope: old candidates only; no segments, no sentiment, no aggregation, no pipeline rewrite
- files planned: new `scripts/run_phase4_step3_residual_hdbscan_diagnostic.py` + plan file updates
- metric: cluster count, clustered share, top-20 manual labels, final verdict keep vs kill
- fallback: at most one softer rerun if base clustering is clearly poor
- goal: phase4_step3_residual_only_hdbscan
- checked: residual-only HDBSCAN after cleaned `domain_priority` routing; base + one softer fallback only
- got: base=176 clusters, share=0.6463; fallback=285 clusters, share=0.7314
- got: top-20 manual labels = 1 useful, 7 duplicates, 10 noise, 1 mixed, 1 unclear
- did not work: top clusters still dominated by anchor duplicates and residual garbage (`СЂР°Р·`, `СѓР¶Рµ`, `РїРѕСЃР»Рµ`, `РїСЂРё`, object names)
- fixed: only one clearly useful new aspect in top-20 (`РіРѕСЂРєРё / Р°С‚С‚СЂР°РєС†РёРѕРЅС‹`)
- decision: `kill_hdbscan_branch`
- next step: stop this branch; return to anchor/routing cleanup or another non-HDBSCAN direction
- goal: phase4_step4_residual_repair_diagnostic
- checked: baseline cleaned domain_priority residual vs repair_v1 exact_anchor_short_circuit + residual_noise_gate
- got: clean exact anchor leakage 807 -> 0; known bad top-10 terms 7 -> 0
- got: residual raw 17642 -> 16115; residual clean 10253 -> 8739
- got: single-token clean share 78.1% -> 74.7%; sample after repair = 84 useful / 36 unclear / 0 noise
- did not work: residual still mostly context-free one-token object/entity lemmas
- fixed: exact anchor leakage is not the remaining bottleneck
- decision: kill_residual_branch
- next step: do not return to HDBSCAN on isolated candidate lemmas; use this only as diagnostic evidence
- goal: phase4_step5_contextual_residual_hdbscan_probe
- checked: HDBSCAN on repair_v1 residual_clean using context_window_text embeddings, not candidate_lemma
- got: input rows 8249; min_cluster_size=10; min_samples=5; fallback=false
- got: clusters=2; clustered_share=0.6065
- got: top labels = 1 useful_new_aspect, 1 too_mixed
- did not work: one giant mixed cluster absorbed 4993/8249 rows across categories
- fixed: bare-word clustering is not the only issue; short context embeddings also collapse
- decision: kill_contextual_hdbscan_branch
- next step: do not pursue residual HDBSCAN unless changing representation/evaluator, which would be a new architecture variable

- goal: discovery_step1_add_sbert_large_encoder
- checked: new discovery module must stay isolated from the main pipeline
- baseline: old encoder family is weak for residual semantic separation
- hypothesis: `ai-forever/sbert_large_nlu_ru` gives better discovery embeddings
- files: `src/discovery/__init__.py`, `src/discovery/encoder.py`, `tests/test_discovery_encoder.py`
- next step: add lazy CPU encoder, download model once, run shape/norm/similarity tests


- ???? ?????: discovery_step1_add_sbert_large_encoder
- ??? ?????????: ????????? RU encoder ??? discovery ??? ????????? main pipeline
- ??? ??????????: ???????? `DiscoveryEncoder` ?? `ai-forever/sbert_large_nlu_ru`, lazy load, mean pooling, L2 norm, `(N,1024)`
- ??? ?? ?????????: ???????? negative ???? ? sanity-????? ???? ??????? ??????? ??? ???? ??????
- ??? ?????????????: ???????? ???? ????????; cosine similar=0.7956, different=0.4001
- ??? ??????????: `tests/test_discovery_encoder.py` -> `5 passed`
- ????????? ???: ??????? discovery clustering / evaluation ?????? ?????? encoder


- ???? ?????: discovery_step2_residual_phrase_extraction
- ??? ?????????: ?????????? candidate phrases ?? covered ? residual ?? ?????????????????? ??????????? ? vocabulary
- ????????: ??? discovery ??????? lexical residual routing ??? ????? ??????? ? ??? ?????? main pipeline
- ?????: `src/discovery/residual_extractor.py`, `src/discovery/__init__.py`, `tests/test_residual_extractor.py`
- ???????: ?????? `test_residual_extractor.py` ?? mixed / no-residual / fully-residual ??????
- ????????? ???: ??????????? residual extractor ? ???????? unit tests


- ???? ?????: discovery_step2_residual_phrase_extraction
- ??? ?????????: routing candidate phrases ? covered/residual ?? ?????????????????? overlap ? aspect synonyms
- ??? ??????????: ????????? `ResidualExtractor` ? `ResidualResult`
- ??? ??????????: default path ?????????? ???????? `CandidateExtractor`
- ??? ?????????????: ????? alias `physical_goods -> e-commerce` ??? core vocabulary
- ??? ?? ?????????: `tmp_path` ?????? ????? ??-?? ???? ?? system temp
- ??? ?????????????: ????? ?????????? ?? in-memory `Vocabulary`
- ??? ??????????: `tests/test_residual_extractor.py` -> `3 passed`; ??? discovery tests -> `8 passed`
- ????????? ???: ????????????? ???????/???? ?????? `ResidualExtractor` + ?????? encoder

- цель этапа: discovery_step3_review_representation
- что проверяли: review-level embedding только по 
esidual_phrases
- гипотеза: mean по phrase embeddings даст компактное представление отзыва для clustering
- ограничения: без изменений vocabulary/matching/sentiment/clustering
- файлы: src/discovery/representation.py, src/discovery/__init__.py, 	ests/test_representation.py
- проверка: shape, excluded ids, L2 norm, один вызов encoder на весь батч фраз
- следующий шаг: реализовать batched flatten->encode->group pipeline и прогнать unit tests

- цель этапа: discovery_step3_review_representation
- что проверяли: review embedding только по 
esidual_phrases после шага residual extraction
- что получилось: добавлен ReviewRepresentation; flatten всех фраз -> один encoder.encode() -> mean per review -> L2 norm
- что не сработало: проблем в логике не найдено; остался только не-блокирующий warning pytest cache permissions
- что зафиксировано: отзывы без residual исключаются; batch shape (4,1024) на тестовом наборе; общий discovery test suite = 11 passed
- следующий шаг: clustering review embeddings + top phrases per cluster

- цель этапа: discovery_step4_hdbscan_review_clustering
- что проверяли: clustering review embeddings через HDBSCAN
- гипотеза: плотные группы дадут отдельные кластеры, разнородный набор уйдёт в noise
- ограничения: без изменений residual extraction / encoder / vocabulary / sentiment
- файлы: src/discovery/clusterer.py, src/discovery/__init__.py, 	ests/test_clusterer.py
- проверка: synthetic 3-group, all-noise case, корректные cluster stats
- следующий шаг: реализовать ClusteringResult и unit tests

- цель этапа: discovery_step4_hdbscan_review_clustering
- что проверяли: HDBSCAN на review embeddings после ReviewRepresentation
- что получилось: добавлен ReviewClusterer; считаются 
eview_to_cluster, cluster_sizes, 
_clusters, 
_noise, 
oise_rate
- что не сработало: проблем в логике не найдено; остался только не-блокирующий warning pytest cache permissions
- что зафиксировано: synthetic 3-group case проходит; heterogeneous case уходит полностью в noise; общий discovery suite = 13 passed
- следующий шаг: top residual phrases per cluster + evaluation against gold labels

- цель этапа: discovery_step5_cluster_phrase_aggregation
- что проверяли: summary по residual phrases внутри найденных review clusters
- гипотеза: частоты фраз по кластеру дадут интерпретируемый top phrases
- ограничения: без изменений clustering / encoder / residual extraction / vocabulary
- файлы: src/discovery/aggregator.py, src/discovery/__init__.py, 	ests/test_aggregator.py
- проверка: правильный top, -1 не агрегируется, sample review ids <= 5
- следующий шаг: реализовать aggregator и unit tests

- цель этапа: discovery_step5_cluster_phrase_aggregation
- что проверяли: summary по non-noise кластерам из ResidualResult + ClusteringResult
- что получилось: добавлен ClusterAggregator; считаются 	op_phrases и sample_review_ids
- что не сработало: проблем в логике не найдено; остался только не-блокирующий warning pytest cache permissions
- что зафиксировано: cluster_id=-1 не агрегируется; top phrases сортируются по частоте; общий discovery suite = 15 passed
- следующий шаг: gold-based evaluation качества кластеризации

- цель этапа: discovery_step6_gold_cluster_evaluation
- что проверяли: purity и coverage discovery clusters vs gold после удаления словарно-покрытых аспектов
- что получилось: добавлен ClusterEvaluator и EvaluationReport; считаются purity_per_cluster, dominant_aspect_per_cluster, coverage_via_clustering, 
_clean_clusters
- что не сработало: проблем в логике не найдено; остался только не-блокирующий warning pytest cache permissions
- что зафиксировано: synthetic 2-cluster case проходит; all-noise case даёт coverage=0; общий discovery suite = 17 passed
- следующий шаг: end-to-end discovery pipeline / runner по категории

- цель этапа: discovery_step7_pipeline_wrapper
- что проверяли: единый end-to-end wrapper поверх discovery stages
- гипотеза: orchestration-обёртка повысит воспроизводимость запуска без изменения логики этапов
- ограничения: без изменений residual extraction / representation / clustering / aggregation / evaluation
- файлы: src/discovery/pipeline.py, src/discovery/__init__.py, 	ests/test_pipeline.py
- проверка: DiscoveryReport, metadata, smoke test со stub encoder
- следующий шаг: реализовать pipeline wrapper и unit test

- цель этапа: discovery_step7_pipeline_wrapper
- что проверяли: единый end-to-end wrapper поверх discovery stages
- что получилось: добавлены DiscoveryReport и 
un_discovery; wrapper собирает summaries, evaluation, metadata
- что не сработало: проблем в orchestration не найдено; остался только не-блокирующий warning pytest cache permissions
- что зафиксировано: smoke test pipeline проходит; metadata содержит model/date/hdbscan; общий discovery suite = 18 passed
- следующий шаг: category-level runner / запуск на реальных review batches

- цель этапа: discovery_step8_runner_and_artifacts
- что проверяли: category-level runner + export JSON/Markdown/CSV artifacts
- что получилось: добавлен enchmark/discovery/run_discovery.py; параметры вынесены в config.discovery_runner; metadata расширены счётчиками и excluded_review_ids
- что не сработало: полный 4-category запуск не выполнялся в этом шаге из-за тяжёлого CPU runtime на большой модели
- что зафиксировано: helper tests проходят; --help работает; общий discovery suite = 20 passed
- следующий шаг: реальный запуск enchmark/discovery/run_discovery.py и анализ артефактов

- goal: discovery_step8_real_benchmark_run
- checked: full 4-category run with frozen discovery runner config
- got: artifacts saved to `benchmark/discovery/results/20260424_171034`
- got: physical_goods -> residual=950/1094, clusters=4, clean=2, coverage=0.0707, noise=0.6747
- got: consumables -> residual=200/200, clusters=2, clean=0, coverage=0.0000, noise=0.7800
- got: hospitality -> residual=197/197, clusters=0, clean=0, coverage=0.0000, noise=1.0000
- got: services -> residual=270/271, clusters=0, clean=0, coverage=0.0000, noise=1.0000
- did not work: clustering quality is weak; only physical_goods produced any clean clusters, overall coverage stays near zero
- fixed: runner itself works end-to-end and exports JSON/Markdown/CSV artifacts
- decision: run_pass_quality_fail
- next step: inspect `physical_goods` summaries and residual inputs before touching params

- goal: discovery_step9_encoder_sanity_check
- checked: fixed Russian pair sets for `rubert-tiny2` vs `sbert_large_nlu_ru`
- got: artifacts saved to `benchmark/discovery/results/20260424_204200_sanity`
- got: `rubert-tiny2` medians -> similar=0.8407, different=0.6499, random_nouns=0.8217
- got: `sbert_large_nlu_ru` medians -> similar=0.7594, different=0.4343, random_nouns=0.5694
- got: separation for `sbert_large_nlu_ru` = 0.3251
- got: similar/different ranges for `sbert_large_nlu_ru` are not fully nested
- did not work: `rubert-tiny2` still keeps random nouns unnaturally high
- decision: `модель разделяет`
- next step: move to next clustering diagnostic with `sbert_large_nlu_ru`, keeping caution about residual quality rather than encoder collapse

- goal: discovery_step10_product_level_runner
- checked: discovery grouped by product (`nm_id`) instead of category
- changed: runner/artifacts only; residual extraction, encoder, HDBSCAN params, evaluation, vocabulary unchanged
- got: artifacts saved to `benchmark/discovery/results/20260424_210212_by_product`
- got: physical_goods -> products=9, clusters=2, clean=2, weighted_coverage=0.0254, weighted_noise=0.9579
- got: consumables/hospitality/services -> 0 clusters, weighted_noise=1.0000
- got: only product with clusters = `physical_goods/209269133`, coverage=0.2317, noise=0.6154
- did not work: product-level grouping made category-level aggregate worse than previous category run
- fixed: product-level runner and artifacts are reproducible
- verified: discovery suite = 22 passed; pytest cache warning non-blocking
- decision: run_pass_quality_fail
- next step: stop treating HDBSCAN residual discovery as useful under frozen representation/params; inspect residual representation or choose another branch

- goal: discovery_step11_residual_quality_diagnostic
- checked: residual input before clustering
- got: artifacts `benchmark/discovery/results/20260424_212249_residual_quality`
- got: residual unigram_share ~0.63-0.66 across categories
- got: residual_gold_hit_rate varied: consumables 0.0294, hospitality 0.8226, physical_goods 0.4427, services 0.6222
- did not work: top residuals included dictionary-covered delivery/packaging-like terms
- fixed: diagnosed extra domain filtering inside ResidualExtractor as root cause
- next step: isolated residual domain-filter fix

- goal: discovery_step12_residual_domain_filter_fix
- checked: use all aspects from loaded hybrid vocabulary for residual coverage
- changed: ResidualExtractor aspect selection only; no encoder/HDBSCAN/evaluator/vocabulary changes
- got: residual phrases reduced by 766 total; consumables 1771 -> 1489
- got: category benchmark `20260424_212937`; physical_goods coverage 0.0707 -> 0.0727, consumables clusters 2 -> 0
- got: product benchmark `20260424_214335_by_product`; aggregate quality unchanged, only physical_goods/209269133 clusters
- verified: discovery suite = 25 passed; pytest cache warning non-blocking
- decision: fix_pass_quality_still_fail
- next step: kill HDBSCAN residual discovery as final method or run one representation diagnostic before final kill decision

- goal: discovery_v2_phrase_per_product
- checked: per-product clustering of unique residual phrases with weights
- changed: new v2 pipeline/runner only; old pipeline untouched; HDBSCAN min_cluster_size=5, min_samples=3
- got: artifacts `benchmark/discovery/results/20260424_223317_per_product`
- got: 16/16 products produced clusters; mean coverage=0.0046; mean noise=0.6125
- got: non-zero coverage only for services/1526918294=0.0455 and hospitality/1809358565=0.0286
- got: meaningful manual clusters exist for РєР°С‚СѓС€РєР°/СЃРїРёРЅРЅРёРЅРі, С…СѓРґРё/С€РІС‹, С‚Р°СЂР°РєР°РЅС‹/СЌС„С„РµРєС‚, РєРѕСЂРј/РєРѕС‚, С‚СѓР°Р»РµС‚/Р·Р°Р»
- did not work: gold-heuristic coverage remains almost zero; many clusters are numbers/dates/generic words
- fixed: old artifacts removed; new JSON/MD/CSV/summary artifacts created
- verified: discovery-related tests 29 passed; runtime 0:08:37
- decision: run_pass_quality_mixed; useful for exploratory mining, risky for automatic vocabulary expansion
- next step: add residual phrase filtering/ranking before clustering, or use v2 only as manual vocabulary discovery artifact

- goal: discovery_v3_filter_metrics_manual_eval
- checked: plan for isolated A/B on per-product phrase discovery
- baseline: v2 artifacts `20260424_223317_per_product`
- changed: phrase preprocessing + evaluation metrics only
- constraints: do not change encoder/vocabulary/sentiment/matching/old v2 pipeline
- metrics: noise_rate, cohesion, separation, silhouette, concentration, coverage@0.65, soft_purity, novel clusters
- next step: implement new v3 modules, tests, runner, then run A/B

- goal: discovery_v3_filter_metrics_manual_eval
- checked: A/B no_filter vs filtered on per-product phrase discovery
- got: artifacts `benchmark/discovery/results/20260424_231742_v3`
- got: 16 products, 32 rows, runtime 0:16:48
- got: filter removed 4008/21363 phrases = 18.76%
- got: noise_rate improved 0.6125 -> 0.5366
- got: coverage@0.65 improved 0.6600 -> 0.7466; soft_purity 0.1499 -> 0.1706
- did not work: cohesion 0.6994 -> 0.6692; separation 0.5817 -> 0.6056; silhouette 0.2327 -> 0.2185
- fixed: added phrase filter, L1/L2 metrics, manual_eval, v3 runner, manual labels template
- verified: v3 tests 20 passed; discovery explicit suite 49 passed
- note: stale `tests/test_discovery/tmp*` dirs from failed tempfile test have Windows ACL deny; `pytest.ini` excludes tmp* from collection
- decision: RUN PASS / FILTER MIXED; auto verdict says filter hurts, do not replace v2 automatically
- next step: human review of `manual_cluster_labels.csv` before deciding if filtered clusters are qualitatively better

- goal: discovery_gold_aspects_per_product_reference
- checked: unique gold aspect names per `nm_id` for manual cluster labeling
- source: `data/dataset_final.csv`
- got: `benchmark/discovery/gold_aspects_per_product.md`
- got: 16 products, 1762 reviews processed
- got: skipped empty/invalid `true_labels` = 0
- got: missing products = 0
- fixed: product order from requested category_mapping; aspects alphabetic, no normalization/filtering
- next step: use as reference while filling `manual_cluster_labels.csv`

- С†РµР»СЊ СЌС‚Р°РїР°: discovery_v3_final_manual_metrics
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: join СЂСѓС‡РЅРѕР№ L3-СЂР°Р·РјРµС‚РєРё Рє РіРѕС‚РѕРІС‹Рј v3 L1/L2 РјРµС‚СЂРёРєР°Рј
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: РїР»Р°РЅ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅ; РЅСѓР¶РµРЅ manual_cluster_labels_draft.csv
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: draft-С„Р°Р№Р» РїРѕРєР° РЅРµ РЅР°Р№РґРµРЅ РІ benchmark/discovery/manual_labels
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: v3 source dir = benchmark/discovery/results/20260424_231742_v3
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: РґРѕР±Р°РІРёС‚СЊ final runner Рё Р·Р°РїСѓСЃС‚РёС‚СЊ РїРѕСЃР»Рµ РїРѕСЏРІР»РµРЅРёСЏ draft labels

- С†РµР»СЊ СЌС‚Р°РїР°: discovery_v3_final_manual_metrics
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: L3 join manual labels Рє РіРѕС‚РѕРІС‹Рј L1/L2 v3 Р±РµР· РїРµСЂРµСЃС‡С‘С‚Р°
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: output benchmark/discovery/results/20260424_231742_final
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: РЅРµС‚; draft РЅР°Р№РґРµРЅ РІ benchmark/ Рё СЃРєРѕРїРёСЂРѕРІР°РЅ РІ benchmark/discovery/manual_labels
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: 86/86 labeled; valid=53 (61.6%), valid_novel=32, mixed=22, noise=11
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ final_report.md Рё metrics_summary_final.csv РІ Р’РљР 

- goal: final_e2e_pipeline_run
- checked: frozen lexical detection + filtered v3 discovery + fixed NLI formula + Bayesian aggregation
- got: output `benchmark/end_to_end/results/20260425_110116_final_e2e/`, runtime 2388.0 sec
- got: Track A P/R/F1=0.4767/0.4198/0.4279; review MAE=1.1528; product MAE n>=3=0.8755
- got: Track B P/R/F1=0.5698/0.4545/0.4847; review MAE=1.2677; product MAE n>=3=0.9397
- got: Track C star review MAE=0.6398; product MAE n>=3=0.5503
- did not work: sentiment sanity failed vs expected 0.65-0.75; discovery worsened sentiment/product MAE
- fixed: detection close to frozen baseline; no temperature tuning, no LLM, no discovery recompute
- checked: NLI label mapping correct; hard cases show fixed hypothesis-template inversions
- next step: use results honestly; any sentiment repair must be a separate isolated experiment- С†РµР»СЊ СЌС‚Р°РїР°: sentiment_mae_search
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: current/main/origin refs, git history, .opencode artifacts for review-level MAE <=0.5
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: global review-level <=0.5 not found
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: РЅР°Р№РґРµРЅРЅС‹Рµ <=0.5 РѕС‚РЅРѕСЃСЏС‚СЃСЏ Рє category/product/per-product, not global review-level
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: phase3 baseline 0.7116; local sentence 0.6262; candidate code copied to sentiment_search artifact
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: if needed, run isolated A/B old SentimentEngine formula vs final_e2e formula on same matched pairs

- goal: final_e2e_sentiment_engine_v4
- checked: custom final_e2e sentiment -> faad23a v4 batch_analyze, review-level premise
- got: P_ent+P_neutral run Track A MAE=0.9934; exact faad23a P_ent+P_contra run Track A MAE=0.9565
- did not work: sanity vs 0.7116 failed by +0.2449; product MAE stayed high 0.8218
- fixed: detection unchanged 0.4767/0.4198/0.4279; star unchanged 0.6398
- root cause: phase3 0.7116 artifact looks like v5 dual-hypothesis, not faad23a v4; common-pair scores differ strongly
- fixed artifacts: benchmark/end_to_end/results/20260425_154958_final_e2e
- next step: isolated rerun with current/v5 SentimentEngine.batch_analyze if target MAE 0.65-0.75 is required

- goal: final_e2e_negation_absence_correction
- checked: sentiment-only correction for absence-positive negation after NLI; detection/discovery/aggregation unchanged
- got: output `benchmark/end_to_end/results/20260425_165408_final_e2e`, runtime 1789.5 sec
- got: Track A P/R/F1=0.4767/0.4198/0.4279; review MAE=0.8466; product MAE=0.7841
- got: Track B P/R/F1=0.5698/0.4545/0.4847; review MAE=0.9250; product MAE=0.9140
- got: corrections=31/5363, eligible low-high=549, inversion_rate=5.65%, MAE before/after=0.9943->0.9563
- did not work: consumables MAE=0.5342 missed <0.50; correction count below broad target 50-150
- fixed: per_aspect has `negation_correction_applied`; summary has negation stats and sanity block
- next step: separate broader negation experiment only if false-positive risk is reviewed

- С†РµР»СЊ СЌС‚Р°РїР°: discovery_v3_snapshot_reuse
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: РјРѕР¶РЅРѕ Р»Рё РєРµС€РёСЂРѕРІР°С‚СЊ per-product discovery v3 Р±РµР· РёР·РјРµРЅРµРЅРёСЏ clustering
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: РґРѕР±Р°РІР»РµРЅ fingerprint РїРѕ reviews/gold/vocab/config/filter mode Рё JSON snapshot cache
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: РїРѕР»РЅС‹Р№ discovery run РЅРµ Р·Р°РїСѓСЃРєР°Р»СЃСЏ РїРѕ СѓСЃР»РѕРІРёСЋ Р·Р°РґР°С‡Рё
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: runner v3 РїРёС€РµС‚ cache hit/miss Рё reuse РіРѕС‚РѕРІС‹Р№ ProductDiscoveryReport
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: РїСЂРё СЃР»РµРґСѓСЋС‰РµРј СЂРµР°Р»СЊРЅРѕРј discovery РїСЂРѕРіРѕРЅРµ РїСЂРѕРІРµСЂРёС‚СЊ, С‡С‚Рѕ РІС‚РѕСЂРѕР№ Р·Р°РїСѓСЃРє РґР°С‘С‚ cache hits

- С†РµР»СЊ СЌС‚Р°РїР°: traced_pipeline_refactor_v1
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏРµРј: refactor current e2e into traced compute-once artifacts
- baseline: `benchmark/end_to_end/results/20260425_165408_final_e2e`
- РїРµСЂРµРјРµРЅРЅР°СЏ: orchestration/artifact layout only; algorithms frozen
- Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: Stage3 lexical-only; cosine diagnostic; Stage4 frozen v3 centroid cosine threshold=0.5
- gate: MAE review 0.8466, round 0.8005, product n3 0.7841, P/R/F1 0.4767/0.4198/0.4279, inversion 0.0565
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: implement `src/pipeline/` traced runner + evaluation sanity tests

## traced_pipeline_refactor_v1 вЂ” compact
- С†РµР»СЊ СЌС‚Р°РїР°: compute once, analyze N times РґР»СЏ С‚РµРєСѓС‰РµРіРѕ e2e
- С‡С‚Рѕ РїСЂРѕРІРµСЂСЏР»Рё: parity СЃ 20260425_165408_final_e2e Р±РµР· СЃРјРµРЅС‹ Р°Р»РіРѕСЂРёС‚РјРѕРІ
- С‡С‚Рѕ РїРѕР»СѓС‡РёР»РѕСЃСЊ: full run results/20260425_183110_traced СЃРѕР·РґР°РЅ
- С‡С‚Рѕ РЅРµ СЃСЂР°Р±РѕС‚Р°Р»Рѕ: РЅРµС‚; sanity gate РїСЂРѕС€С‘Р»
- С‡С‚Рѕ Р·Р°С„РёРєСЃРёСЂРѕРІР°РЅРѕ: lexical-only, frozen v3 binding threshold 0.5, v4 sentiment, deterministic candidate_id
- Р°СЂС‚РµС„Р°РєС‚С‹: MANIFEST, parquet/npy/csv/json, figures, dashboard screenshots
- С‚РµСЃС‚С‹: tracing/unit/sanity/core passed
- СЃР»РµРґСѓСЋС‰РёР№ С€Р°Рі: Р°РЅР°Р»РёР· Р’РљР  РЅР° traced artifacts
- goal: sentiment_benchmark_evidence_modes_v1
- checked: 4 isolated sentiment modes on frozen traced artifacts; no clustering rerun
- got: A MAE=0.9770 cov=0.4489; B=0.8878 cov=0.3163; C=0.8934 cov=0.3181; D=0.9014 cov=0.3483
- got: best review MAE = B
- got: best discovery-pair MAE = C (0.9262)
- got: D improves coverage vs B/C but stays worse on MAE
- did not work: discovery evidence is incomplete; 2053 assigned discovery aspects have no recoverable phrase hit in traced artifacts
- fixed: new benchmark/sentiment scaffold with per-mode runners, metrics, hard cases, and saved outputs
- verified: py_compile OK; pytest 3 passed; smoke A/B/C/D passed; full A/B/C/D runs saved
- artifacts: benchmark/sentiment/mode_a_current_baseline/results/20260430_174630
- artifacts: benchmark/sentiment/mode_b_sentence_evidence/results/20260430_175836
- artifacts: benchmark/sentiment/mode_c_window_evidence/results/20260430_180307
- artifacts: benchmark/sentiment/mode_d_multi_evidence/results/20260430_181922
- next: optional D weighted_relevance A/B; primary follow-up candidate = B
- goal: sentiment_mode_d_weighted_relevance_v1
- checked: final sentiment-only D rerun with weighted aggregation on frozen traced artifacts
- got: output benchmark/sentiment/mode_d_multi_evidence_weighted_relevance/results/20260501_103909
- got: review MAE 0.9014 -> 0.8892 vs D; round 0.8834 -> 0.8707; vocab pair 0.8573 -> 0.8456
- got: discovery pair 1.0305 -> 1.0223; coverage unchanged 0.3483; kept_after_threshold unchanged 3578
- did not work: B still keeps best primary review MAE 0.8878 and much better discovery pair MAE 0.9611
- fixed: separate mode_id + runner for D_weighted; no extraction/matching/vocabulary/discovery changes
- verified: full run completed; runtime 1084.28 sec
- artifacts: benchmark/sentiment/reference_summary.md
- decision: keep B as reference
- fallback: D_weighted only if higher localized-evidence coverage matters more than best MAE

- goal: final_res_v2_sentence_evidence
- checked: traced full-run launcher and stable final result aliases
- got: entrypoint = `python -m src.pipeline.run_traced_pipeline --config run_config.yaml`
- got: code path = `src/pipeline/run_traced_pipeline.py` -> `src/pipeline/orchestrator.py::run_traced_pipeline`
- got: created `results/final_res_v1` from frozen `results/20260425_183110_traced`
- got: created `results/final_res_v2` with reused S1-S4 and rebuilt S5-S6 from mode B
- got: created `benchmark/manual_audit/final_v2`
- got: `final_res_v1` Track A review MAE `0.8466`; `final_res_v2` Track A `0.8616`; `final_res_v2` Track B `0.8878`
- did not work: `final_res_v2` does not beat `final_res_v1` on Track A vocab-only MAE
- fixed: added `scripts/freeze_final_results.py`, README note, test `tests/test_freeze_final_results.py`
- verified: `pytest tests/test_freeze_final_results.py -q --basetemp .pytest_tmp_manual` -> `2 passed`

- goal: final_res_v1_vs_v2_comparison
- checked: frozen `final_res_v1` vs `final_res_v2` on saved artifacts only
- got: review MAE `0.8466 -> 0.8616`; round `0.8005 -> 0.8479`; vocab pair `0.8274 -> 0.8466`
- got: discovery pair MAE `1.0686 -> 0.9611`; coverage `0.2249 -> 0.3163`
- got: product MAE n3 nearly flat `0.7841 -> 0.7851`; star baseline unchanged
- got: category winners = `consumables`, `services`; regressions = `physical_goods`, slight `hospitality`
- got: per-product review MAE improved on `9/16`, worsened on `7/16`
- got: hard-cases count unchanged `30`; source mix `discovery 24 -> 10`, `vocab 6 -> 20`
- did not work: `final_res_v2` did not beat `final_res_v1` on main vocab-only review metrics
- fixed: commit `2d89dce` saves freeze builder/test/manual-audit summary
- decision: keep `final_res_v1` as safer final package; use `final_res_v2` as higher-coverage sentence-evidence variant
- goal: final_res_v1_vs_v2_sentiment_diagnostics_v1
- checked: richer pair-level comparison for frozen `final_res_v1` vs `final_res_v2`
- got: added metrics script `scripts/compare_final_sentiment_runs.py`
- got: wrote `sentiment_metrics_comparison.csv`, `sentiment_metrics_by_product.csv`, `sentiment_metrics_by_category.csv`
- got: own-pairs `v2` better on MAE `0.9563 -> 0.8953`, round MAE `0.9222 -> 0.8761`, `Acc@1` `0.6712 -> 0.7259`
- got: common-pairs `v2` also better on MAE `0.9197 -> 0.8986` and `Acc@1` `0.6967 -> 0.7222`
- got: `v2` worse on `RMSE` `1.4236 -> 1.5280`, strong wrong-polarity `0.1148 -> 0.1501`, direction accuracy `0.7595 -> 0.7521`
- got: real trade-off = `vocab` worse `0.8274 -> 0.8466`, `discovery` better `1.0686 -> 0.9611`
- got: `nm_id=619500952` systemic collapse, not one outlier: `MAE 1.03 -> 2.22`, bias `-0.94 -> -2.14`, wrong-polarity `0.25 -> 0.51`
- got: hard-case source mix shifted from mostly `discovery` to mostly `vocab`
- verified: `pytest tests/test_compare_final_sentiment_runs.py -q --basetemp .pytest_tmp_manual` -> `3 passed`
- decision: keep `final_res_v1` as safer default; `v2` numerically closer but less safe on catastrophic polarity flips
- goal: sentiment_benchmark_abcd_diagnostics_v1
- checked: saved A/B/C/D benchmark artifacts only; no new inference runs
- got: script `scripts/compare_sentiment_benchmark_modes.py`
- got: output `benchmark/sentiment/mode_abcd_diagnostics/results/20260501_130143`
- got: own-pairs MAE ranking `C 0.8851 < B 0.8953 < D 0.9373 < A 0.9943`
- got: Acc@1 ranking `C 0.7408 > B 0.7259 > D 0.7132 > A 0.6612`
- got: coverage ranking `A 0.4489 > D 0.3483 > C 0.3181 > B 0.3163`
- got: common-pairs ranking stayed same: `C`, `B`, `D`, `A`
- got: `B` best vocab MAE `0.8466`; `C` best discovery MAE `0.9262`
- got: `A` best RMSE `1.4760`, but weak MAE/Acc@1 because full-review baseline is conservative and coverage-heavy
- got: `D` keeps more localized pairs than `B/C`, but loses on MAE and strong wrong-polarity
- verified: `pytest tests/test_compare_sentiment_benchmark_modes.py -q --basetemp .pytest_tmp_manual` -> `2 passed`
- decision: `B` still strongest reference; `C` best pure numeric mode; `D` only for coverage-first fallback
- goal: final_res_v2_window_evidence
- checked: rebuild stable `final_res_v2` from saved mode C artifacts only
- got: generalized `freeze_final_results.py` from hardcoded B to configurable mode
- got: `results/final_res_v2` now points to `mode_c_window_evidence`
- got: Track A worsened vs `final_v1`: review MAE `0.8466 -> 0.8696`, round `0.8005 -> 0.8528`, product n3 `0.7841 -> 0.8703`
- got: Track B improved vs `final_v1`: review MAE `0.9250 -> 0.8934`, discovery pair `1.0686 -> 0.9262`
- got: pair-level own-pairs improved strongly: MAE `0.9563 -> 0.8851`, `Acc@1` `0.6712 -> 0.7408`, median AE `0.5704 -> 0.1267`
- got: safety worsened: wrong-polarity `0.1761 -> 0.1894`, strong wrong-polarity `0.1148 -> 0.1563`, RMSE `1.4236 -> 1.5380`
- verified: `pytest tests/test_freeze_final_results.py -q --basetemp .pytest_tmp_manual` -> `3 passed`
- artifacts: `results/final_res_v2`, `benchmark/manual_audit/final_v2`, `results/final_res_v1_vs_v2_diagnostics/20260501_132850`
- decision: keep `final_v1 = A`, set `final_v2 = C`; use `v2` as stronger numeric alternative, not as safer replacement
- goal: test_end_to_end_honest_ab_demo_v1
- checked: minimal honest e2e on 20 hardcoded reviews
- got: new folder `test_end_to_end/`
- got: saved entities `reviews`, `aspects`, `fragments`, `aspect_assignments`
- got: restored `mode_a`, `mode_b`, `mode_c` only from saved artifacts
- got: 20 reviews, 47 fragments, 47 assignments
- got: `A/B/C` all restored 47 inputs each
- got: no repeated search of aspect text during restoration
- verified: `.venv\Scripts\python.exe test_end_to_end\demo_pipeline.py`
- artifacts: `test_end_to_end/generated/*`
- decision: this is the target honest data model for future real A/B sentiment runs
- цель этапа: sentiment_assignment_freeze_full_run
- что проверяли: честная общая база aspect-review assignments для full run и A/B/C
- что получилось: traced run теперь пишет `aspect_review_assignments.parquet` и `discovery_candidate_bindings.parquet`
- что получилось: benchmark A/B/C строятся из одного списка assignments, не из разных путей
- что не сработало: старый `candidate_id` коллидировал на одинаковом lemma+start
- что исправлено: `candidate_id = stable_id(review_id, lemma, start, end)`
- что зафиксировано: smoke run `20260501_191353_traced`; `discovery_without_evidence=0`; `len(A)=len(B)=len(C)=895`
- следующий шаг: пересчитать benchmark-метрики на новых traced artifacts и только потом выбирать final mode
- цель этапа: sentiment_assignment_model_fix_v2
- что проверяли: разделение review-aspect assignments и evidence fragments в full run
- что получилось: теперь пишутся `aspect_review_assignments.parquet` и `aspect_review_evidence.parquet`
- что не сработало: первая версия схемы раздула assignments до candidate-level (`24475` вместо `6224` уникальных review-aspect)
- что зафиксировано: smoke run `20260501_234150_traced`; `A=459`, `B=459`, `C=459`, `D=895`; `discovery_without_evidence=0`
- тесты: `tests/test_sentiment_benchmark_common.py` -> `5 passed`
- следующий шаг: полный честный rerun A/B/C/D/D_weighted на новой схеме
- goal: sentiment_honest_abcd_weighted_fullrun_v1
- checked: full honest rerun A/B/C/D/D_weighted on shared `review-aspect` assignments after schema fix
- got: traced run `20260502_111627_traced` with `6224` assignments and `24475` evidence, all modes built from same assignment base
- got: result dirs `A 20260502_091441`, `B 20260502_093932`, `C 20260502_095206`, `D 20260502_104703`, `Dw 20260502_114752`
- got: comparison `benchmark/sentiment/mode_abcd_diagnostics/results/20260502_114842`
- got: best own-pairs MAE = `B 0.9256`; best common-pairs MAE = `C 0.8960`; best localized coverage = `D/Dw 0.4194`
- got: weighted D beats plain D (`own 0.9885 -> 0.9627`, `common 0.9147 -> 0.9041`) but still trails B/C
- got: A remains safest on RMSE / strong wrong-polarity, but loses on MAE and Acc@1
- decision: keep honest reference = `B`; keep `C` as best common-pair / fastest localized variant; keep `D_weighted` only as coverage-oriented fallback
- next: if needed, freeze final package only after deciding whether priority is own-pairs MAE (`B`) or common-pair closeness (`C`)

- goal: sentiment_global_cache_v1
- checked: persistent sqlite NLI cache for full run and benchmark
- got: shared cache backend for v4 and v5; stored value = raw logits
- got: cache key = model_signature + premise_hash + hypothesis_hash
- got: full traced summary now reports nli_cache stats
- got: benchmark summary now reports nli_cache stats
- verified: new cache tests + existing sentiment tests = 14 passed
- verified: traced smoke run warm cache -> persistent_hits=458, misses=0
- verified: benchmark mode C warm cache -> persistent_hits=914, misses=0
- fixed: repeated reruns now add only unseen NLI pairs
- decision: honest A/B remains honest; cache changes runtime only
- next: use same cache for future final mode reruns
- goal: fullrun_baseline_a_cache_freeze_v1
- checked: full baseline-A traced run twice on a new empty persistent cache
- got: cold run `20260502_164247_traced`, warm run `20260502_171530_traced`
- got: frozen cache `cache/nli_global_frozen_fullrun_20260502.sqlite3`
- got: elapsed `1925.1165 -> 895.8639 sec` (`-53.46%`)
- got: cold cache `hits=0 misses=6217 writes=6215`; warm cache `hits=6217 misses=0 writes=0`
- got: cache file size `6094848` bytes; no wal/shm leftovers
- got: Track A and Track B metrics identical across runs
- got: hashes identical for `nli_predictions`, `product_aggregates`, `aspect_review_assignments`, `aspect_review_evidence`, `candidate_matches`, `candidates`
- fixed: baseline A now has a frozen reusable global NLI cache for future tests
- decision: keep variant A as full-run baseline reference
- next: reuse this cache in future sentiment/full-pipeline reruns instead of cold recomputation
- goal: stage_cache_s1_s4_v1
- checked: persistent cache for traced `s1-s4` plus existing NLI cache in `s5`
- got: new `src/pipeline/stage_cache.py` and restore helpers for `s1/s2/s3/s4`
- got: `run_summary.json` and `MANIFEST.json` now expose `stage_cache`
- got: root doc `CACHE_WORKFLOW.md`
- verified: unit tests `11 passed`, py_compile OK
- verified: smoke cold/warm on `limit-products=1`
- got: cold `145.0399 sec`, warm `3.0485 sec`
- got: `s1-s4` all `miss -> hit`; `s5` NLI `persistent_hits 0 -> 458`
- got: identical hashes for `candidates`, `candidate_matches`, `aspect_review_assignments`, `nli_predictions`, `product_aggregates`
- decision: keep `stage_cache.enabled = true` by default
- next: future full reruns should reuse both `cache/pipeline_stages` and `cache/nli_global.sqlite3`


- goal: manual_recalc_ui_v1
- checked: separate root-level manual recalculation module on top of traced artifacts only
- got: new `manual_recalc/` with Streamlit card UI, SQLite drafts, batch workflow, AI draft import, prompt copy, CSV/metrics export
- got: review view = `full_text + gold + system + evidence + premise/hypothesis`
- got: persisted tables = `system_decisions`, `gold_decisions`, `review_status`
- got: manual metrics = `precision/recall/F1` + matched-pair `MAE` in strict/soft modes
- verified: `py_compile` OK; smoke load on `results/20260502_171530_traced` -> `1659` reviews, evidence linked
- fixed: duplicate `candidate_id` in traced candidates handled by dedupe in loader
- decision: keep this logic isolated from main dashboard/pipeline
- next: run first real annotation batch and polish prompt / workflow from usage


- goal: manual_recalc_prompt_template_v2
- checked: prompt-layer only for AI draft prefill
- got: replaced template with stricter JSON schema + mapping/sentiment rules
- got: explicit anti-overmapping rule for `????????`
- fixed: no parser/UI/pipeline changes, only prompt wording
- next: test next AI batch draft against the updated template
- goal: manual_recalc_batch_progress_v1
- checked: batch selector visibility for already annotated batches
- got: added batch-level summary from review_status only
- got: labels now show done/partial/new with done/total and draft count
- got: current batch top metric now shows progress, not only size
- did not change: storage schema, metrics logic, pipeline
- verified: pytest test_manual_recalc_batch_progress -> 2 passed
- verified: py_compile OK
- real DB sample: batch_001 partial 23/25 draft 2; batch_006 new
- fixed: annotated batches are now visible in sidebar
- next: reopen Streamlit UI and verify readability on real annotation flow
- goal: manual_recalc_next_batch_button_v1
- checked: one-click batch switch in manual_recalc sidebar
- got: added button `Следующий батч` under `Пачка`
- got: click moves `batch_index` to next with wrap-around
- got: `review_pointer` resets to first review of new batch
- did not change: storage schema, metrics logic, pipeline
- fixed: no need to open batch selectbox for sequential navigation
- next: verify flow in active Streamlit session
- goal: manual_recalc_batch_filter_and_selectbox_fix
- checked: 24/25 after commit + status filter mismatch + batch selectbox double-click
- got: review_status synced every run; invalidate init cache after commit; batch selectbox keyed + clamp
- got: review dropdown keyed per batch_index; status_map keys str(); metric done in-filter vs global caption
- fixed: stale session overwrote committed done; misleading global done denominator under active filter
- goal: manual_recalc_prompt_copy_and_filter_fix_v2
- checked: batch prompt copy still unstable; need single-review prompt + filter consistency
- got: replaced JS clipboard path with download + text_area copy workflow for batch prompt
- got: added single-review prompt export/copy block
- got: fixed FP/FN helper lookups to use str(review_id) keys
- verified: py_compile OK; pytest test_manual_recalc_batch_progress -> 2 passed
- goal: manual_audit_recompute_v1
- checked: manual DB consistency vs traced run 20260502_171530 and dataset_final.csv
- got: new script `scripts/recompute_manual_audit_metrics.py`
- got: saved reports in `manual_recalc/exports/manual_metrics_20260504_184647`
- validation failed 4/8 rules
- missing system rows: 261 across 49 review_ids
- missing gold rows: 89 across 49 review_ids
- FOUND without matched system id: 175; unknown system decisions: 7
- manual detection: P_strict 0.5308, P_soft 0.5798, R 0.6343, F1_strict 0.5780, F1_soft 0.6058
- manual sentiment TP-pair MAE 0.9833, acc@1 0.6197, wrong_polarity 0.1330
- auto Track B: P/R/F1 0.5698/0.4545/0.4847, review MAE 0.9250
- next: cite reports with explicit caveat about incomplete saved audit rows
- goal: manual_audit_repair_v1
- checked: 49 fully missing review audits, 175 FOUND without matched system id, 7 blank system decisions
- got: script `scripts/repair_manual_audit_db.py`
- got: `repair_batch` support in manual_recalc UI via `app_meta`
- got: restored all 49 review rows in sqlite; fixed all 7 blank system decisions
- got: auto-filled 175 FOUND matches; no downgrades needed
- verified: pytest test_manual_recalc_batch_progress -> 3 passed
- verified: py_compile OK for app/storage/repair script
- verified: recompute export `manual_recalc/exports/manual_metrics_20260504_191713`
- fixed: validation now passes 8/8; no missing rows; no unknown statuses
- final: P_strict 0.5194, P_soft 0.5649, R 0.6371, F1_strict 0.5723, F1_soft 0.5989
- final: TP-pair sentiment MAE 0.9771, acc@1 0.6232, wrong_polarity 0.1318
- next: use repaired export for VKR; optional visual QA in `repair_batch`
- goal: manual_vector_sentiment_v1
- checked: fuzzy-vector sentiment eval on repaired manual TP pairs only
- got: rating->vector interpolation for integer and fractional scores on [1..5]
- got: saved `vector_sentiment_metrics.csv`, `vector_sentiment_by_category.csv`, `vector_sentiment_by_product.csv`, `vector_sentiment_summary.md`
- got: figures in `vector_sentiment_figures/`
- verified: `tests/test_recompute_manual_audit_metrics.py` -> `5 passed`; py_compile OK
- overall: L1=0.8343, L2=0.5735, cosine=0.6719, dominant_acc=0.6563
- overall: neutral_collapse=0.3082, polarity_flip=0.0596
- source: `vocab` better than `discovery` on L1/cosine/dominant_acc
- category: best `consumables`, worst `hospitality`
- fixed: no pipeline/inference changes; evaluation layer only
- next: use export `manual_recalc/exports/manual_metrics_20260504_194659`
- цель этапа: handoff_readme_state_v1
- что проверяли: можно ли собрать один короткий handoff по финальному состоянию без новых прогонов
- что получилось: создан `00_README_STATE.md` с `final_main = mode A / full review baseline`
- что получилось: `B/C/D/D_weighted` отмечены как ablation-only, не как финал
- что получилось: manual audit validation зафиксирован как clean, `0/8 failed`
- что получилось: финальные manual metrics перенесены из `manual_metrics_20260504_191713`
- что не сработало: отдельный файл `deep research report` в репо не найден; в handoff помечен как внешний источник обзора аналогов
- что зафиксировано: отрицательные ветки HDBSCAN, contextual HDBSCAN, localized sentiment внесены в handoff
- следующий шаг: использовать `00_README_STATE.md` как опорный state-файл при написании ВКР

- цель этапа: sentiment_postprocess_calibration_v1_run
- что проверяли: влияние формулы постобработки на итоговую sentiment-оценку на сохранённых TP-парах
- что получилось: полный run `benchmark/sentiment_postprocess_calibration/results/20260505_081437`
- что получилось: dry-run best `F9_center_expansion_gamma1_6` -> MAE `0.8878` vs baseline `0.9771`
- что получилось: supervised best `HuberRegressor + with_review_rating` -> MAE `0.6750`, Acc@1.0 `0.7586`
- что не сработало: dual формулы не проверены полноценно, `neg_*` вероятности отсутствуют (all null)
- что зафиксировано: build_dataset merge-fix для `premise_text`; `report.py` восстановлен для генерации summary
- следующий шаг: либо восстановить `neg_*` в traced NLI, либо зафиксировать single-pos+supervised как рабочий постпроцесс

- цель этапа: sentiment_postprocess_negative_completion_v1
- что проверяли: изолированное дозаполнение `neg_*` на тех же 2776 TP строках без изменения pos/ID/text
- что получилось: создан `calibration_dataset_with_dual_nli.csv` в `.../results/20260505_081437`
- что получилось: инварианты пройдены (2776 строк, 0 дублей, 0 null в neg_*, pos_* unchanged, id/review/premise unchanged)
- что получилось: dry-run на dual датасете выполнен, `available_formulas=25` (F2..F8 доступны)
- что не сработало: ничего критичного; completion runtime ~8 мин на CPU
- что зафиксировано: лучший dry-run `F6_dual_logratio_T0_7`, MAE `0.7537`
- следующий шаг: при необходимости выбрать формулу-кандидат и сравнить guard-метрики vs baseline `F0_current`

- goal: sentiment_postprocess negative-only completion + honest dual recalc
- checked: saved TP review/aspect pairs, negative-only NLI fill, dry-run/supervised on same dual dataset
- got: fresh run `20260505_093500`; dual csv has 2776 rows and full neg_* without nulls
- got: dry-run best = F6_dual_logratio_T0_7, MAE 0.7537 vs F0 0.9771
- got: neutral collapse dropped 0.1162 -> 0.0237
- did not work: best dry-run raised wrong polarity 0.1318 -> 0.1491 and slightly worsened RMSE
- fixed: supervised script now uses neg_* + derived dual features by default when dual csv exists
- got: supervised best = Huber + review_rating, MAE 0.6591, RMSE 1.0776, wrong_polarity 0.1081
- fixed: negative runner no longer hardcodes 2776 rows; validates required columns instead
- next: treat dry-run dual formulas as diagnostic unless safe-criterion is satisfied
