# Manual Audit Final Summary

- generated_at: 2026-05-04T19:46:59.927652+00:00
- validation_failed_rules: 0 / 8

## Coverage

1. checked reviews: 1659
2. stored system aspects: 6224 (expected 6224)
3. stored gold aspects: 4389 (expected 4389)

## Manual Detection

4. TP=2776, FP=2138, FN=1581, UNCLEAR=431, DUPLICATE=879
5. manual_precision_strict=0.5194
6. manual_precision_soft=0.5649
7. manual_recall=0.6371
8. manual_f1_strict=0.5723
9. manual_f1_soft=0.5989

## Manual Sentiment

10. matched TP pairs used for sentiment: 2776
11. manual_sentiment_mae=0.9771
12. manual_sentiment_mae_round=0.9496
13. manual_accuracy_at_1_0=0.6232
14. manual_wrong_polarity_rate=0.1318

## Main Error Types

- gold_status: FOUND = 2782 (0.6339)
- gold_status: FN = 1581 (0.3602)
- gold_status: UNCLEAR = 26 (0.0059)
- sentiment_error_type: near_miss = 1730 (0.6232)
- sentiment_error_type: too_low = 410 (0.1477)
- sentiment_error_type: strong_wrong_polarity = 229 (0.0825)
- sentiment_error_type: too_high = 200 (0.0720)
- sentiment_error_type: wrong_polarity = 137 (0.0494)
- sentiment_error_type: large_too_low = 50 (0.0180)
- sentiment_error_type: large_too_high = 20 (0.0072)
- system_manual_decision: TP = 2776 (0.4460)
- system_manual_decision: FP = 2138 (0.3435)

## Manual vs Auto

- auto detection precision (track_b): 0.5698
- auto detection recall (track_b): 0.4545
- auto detection f1 (track_b): 0.4847
- auto sentiment mae review (track_b): 0.9250
- auto sentiment mae round (track_b): 0.8856
- note: auto sentiment in run_summary is review-level Track B; manual sentiment here is TP-pair-level after manual mapping, so MAE values are informative but not strictly the same unit.

## Vkr Conclusion

- Простыми словами: ручная проверка показывает, где модель реально попадает в аспект, а где ошибается на ложных аспектах, дублях и тональности.
- Эти manual-метрики можно использовать в ВКР как честную пост-оценку текущего inference без нового прогона модели.
- Если validation report содержит FAIL, это нужно явно оговорить: часть audit-таблиц заполнена неполно, и итоговые manual-метрики считаются по фактически сохранённой ручной разметке.
