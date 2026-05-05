# Vector Sentiment Summary

- generated_at: 2026-05-04T19:47:02.220527+00:00
- matched_tp_pairs: 2776
- reviews_covered: 1531

## Overall

- mean_vector_l1_distance: 0.8343
- mean_vector_l2_distance: 0.5735
- mean_cosine_similarity: 0.6719
- dominant_class_accuracy: 0.6563
- neutral_collapse_rate: 0.3082
- polarity_flip_rate: 0.0596
- intensity_underestimate_rate: 0.0000

## Rate Denominators

- n_gold_strict_polar_pairs: 2132
- n_gold_strong_polar_pairs: 2132
- n_same_polar_direction_pairs: 1348
- neutral_collapse_rate = collapse / strong gold polarity pairs
- polarity_flip_rate = flips / gold strict polarity pairs
- intensity_underestimate_rate = severe underestimates / same-polarity direction pairs

## Category Highlights

- best_l1_category: consumables (0.6714)
- worst_l1_category: hospitality (0.9038)
- max_polarity_flip_category: hospitality (0.0922)

## Aspect Source

- discovery: l1=0.8949, cosine=0.6432, dominant_acc=0.6192
- vocab: l1=0.7435, cosine=0.7149, dominant_acc=0.7120

## Figures

- vector_sentiment_figures/vector_distance_distributions.png
- vector_sentiment_figures/vector_metrics_by_category.png
- vector_sentiment_figures/vector_metrics_by_aspect_source.png

## Hard Examples

- review_id=00NhRiMR31BSX73L4WLz, gold=Поедаемость:5.00, pred=кот:3.00, l1=2.0000
- review_id=0GxwnOG665W1tQLEgDfn, gold=Запах:5.00, pred=Запах:2.01, l1=2.0000
- review_id=101VbtTD1YKllxLo4UJA, gold=Качество:5.00, pred=кот:1.22, l1=2.0000
