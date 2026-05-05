"""
Ручной маппинг и Manual Precision для 3 held-out товаров.
Решения по каждому аспекту обоснованы ниже.

Запуск: python manual_eval_holdout.py \
    --per-review suite_eval_per_review.json \
    --csv merged_checked_reviews.csv
"""

import json, csv, sys, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

# =========================================================================
# 1. MANUAL MAPPING: predicted_aspect → true_aspect (or None)
#    Логика: смотрим keywords кластера и решаем, какому true аспекту
#    он соответствует. None = нет соответствия в разметке.
# =========================================================================

MANUAL_MAPPING = {
    # ── Гречка (15430704) ──────────────────────────────────────────────
    # True: Качество(88), Логистика(13), Сервис(5), Соответствие(1),
    #       Спам(1), Упаковка(52), Цена(8)
    15430704: {
        # --- anchor-named ---
        "Удобство":       None,           # нет true-эквивалента "удобство готовки"
        "Поедаемость":    "Качество",     # вкус/съедобность гречки = качество еды
        "Здоровье":       None,           # питание/пищеварение — нет в разметке
        "Качество":       "Качество",
        "Продавец":       "Сервис",       # производитель/продавец = сервис
        "Состояние":      "Соответствие", # "увелка плохо", "уракована" = несоответствие
        "Органолептика":  "Качество",     # вкус/запах гречки = качество еды
        "Упаковка":       "Упаковка",
        "Вместимость":    None,           # keywords мусор: "местных тут", "беру"
        "Цена":           "Цена",
        "Логистика":      "Логистика",
        "Свежесть":       "Качество",     # сроки годности = качество для еды
        "Запах":          "Качество",     # запах гречки = качество
        # --- residual medoid-named ---
        "зелёную гречку":   None,         # название разновидности, не аспект
        "гречка":           None,         # название продукта
        "увелки":           None,         # бренд
        "продукт":          None,         # мета-слово
        "круп":             None,         # категория продукта
        "горьковатый вкус": "Качество",   # вкус = качество для еды
        "2500":             None,         # число (вес/цена), мусор как аспект
        "годности хорошие": "Качество",   # срок годности = качество
        "проблем":          None,         # слишком размытое
        "пупырку":          None,         # мусор
    },

    # ── Карты Таро (54581151) ──────────────────────────────────────────
    # True: Внешний вид(82), Инструкция(38), Качество(95),
    #       Комплектация(10), Логистика(12), Соответствие(4),
    #       Упаковка(61), Цена(19)
    54581151: {
        # --- anchor-named ---
        "Удобство":          None,            # для карт: тасование? слишком размыто, keywords пустые
        "Качество":          "Качество",
        "Органолептика":     None,            # запах/вкус нерелевантны для карт
        "Внешний вид":       "Внешний вид",
        "Упаковка":          "Упаковка",
        "Комфорт":           "Качество",      # тактильные ощущения (ламинация, плотность) = качество карт
        "Продавец":          None,            # нет в разметке
        "Состояние":         "Качество",      # дефекты = качество
        "Цена":              "Цена",
        "Вместимость":       None,            # мусор: "пространства", "уголок"
        "Функциональность":  None,            # "энергия", "сила" — эзотерика, нет в true
        "Логистика":         "Логистика",
        "Соответствие":      "Соответствие",
        "Содержание":        "Инструкция",    # сюжеты, тематика, издательство = гайдбук
        "Функционал":        None,            # дубль Функциональности
        "Запах":             None,            # нерелевантен
        # --- residual medoid-named ---
        "формата":              "Соответствие",  # оригинал vs реплика = соответствие
        "хорошие цвета":        "Внешний вид",
        "плотные карты":        "Качество",      # плотность = качество карт
        "покрытие":             "Качество",      # глянец, ламинация = качество
        "коробочка":            "Упаковка",
        "рук":                  None,            # мусор
        "сотрудникам":          None,            # мусор
        "толстые":              "Качество",      # толщина карт = качество
        "вмчтенка":             None,            # непонятная аббревиатура/опечатка
        "дефект":               "Качество",      # дефекты = качество
        "ламинации":            "Качество",      # ламинация = качество
        "картинки":             "Внешний вид",
        "приятная колода":      None,            # общее впечатление, не аспект
        "товар":                None,            # мета
        "ощущениям":            None,            # слишком размыто
        "толкование":           "Инструкция",    # толкование карт = гайдбук/инструкция
        "колода замечательная": None,            # общее впечатление
        "эта колода":           None,            # мета
        "печать":               "Качество",      # качество печати
    },

    # ── Средство от тараканов (619500952) ──────────────────────────────
    # True: Безопасность(26), Запах(26), Использование(22),
    #       Комплектация(7), Логистика(2), Расход(10), Следы(1),
    #       Упаковка(4), Цена(9), Эффективность(100)
    619500952: {
        # --- anchor-named ---
        "Качество":          "Эффективность",  # "качество средства" = насколько эффективно
        "Удобство":          "Использование",  # удобство = простота использования
        "Состав":            None,             # keywords "отдельные особи" — мусор
        "Упаковка":          "Упаковка",
        "Комфорт":           None,             # "большую поверхность" — не про комфорт
        "Здоровье":          "Безопасность",   # здоровье ≈ безопасность для людей/питомцев
        "Функциональность":  "Эффективность",  # эффект, задача = эффективность
        "Содержание":        None,             # "свою историю" — мусор, нерелевантно
        "Продавец":          None,             # нет в разметке
        "Вместимость":       None,             # "места", "этаже" — про квартиру, не аспект
        "Органолептика":     "Запах",          # запах средства
        "Состояние":         None,             # "хулиганов", "трупов" — про тараканов
        "Цена":              "Цена",
        "Логистика":         "Логистика",
        "Поедаемость":       None,             # абсурд для средства от тараканов
        "Свежесть":          None,             # нерелевантно
        "Запах":             "Запах",
        "Функционал":        "Эффективность",  # дубль функциональности
        # --- residual medoid-named ---
        "пор тараканов":     None,             # мета про тараканов
        "отличное средство": None,             # общее впечатление
        "обработки":         "Использование",  # процесс обработки = использование
        "питомцев":          "Безопасность",   # кошка, собаки = безопасность для питомцев
        "ловушками":         None,             # про другие средства (ловушки)
        "засилье":           None,             # мусор
        "ванной":            None,             # локация, не аспект
        "углы":              None,             # локация
        "перестраховки":     "Безопасность",   # респиратор, защита = безопасность
        "столешницу":        None,             # локация
        "эффекте":           "Эффективность",  # прямое совпадение
        "квартиру":          None,             # локация
        "рыжих насекомых":   None,             # мета про тараканов
        "применения":        "Использование",  # применение = использование
        "два-три":           None,             # "2-3 дня" — таймфрейм, слишком размыто
    },
}


# =========================================================================
# 2. MANUAL PRECISION: TP / FP для каждого аспекта
#    Вопрос: "Этот аспект реально релевантен для данного товара?"
#    (независимо от наличия в true_labels)
# =========================================================================

# "anchor" = пришёл из якорного словаря, "residual" = из HDBSCAN medoid
MANUAL_PRECISION = {
    # ── Гречка ─────────────────────────────────────────────────────────
    15430704: {
        # anchor-named
        "Удобство":       "TP",   # удобство приготовления — обсуждается
        "Поедаемость":    "TP",   # вкусовые качества гречки — обсуждается
        "Здоровье":       "TP",   # питание, аллергия, пищеварение — обсуждается
        "Качество":       "TP",
        "Продавец":       "TP",   # производитель, продавец
        "Состояние":      "TP",   # состояние продукта при получении
        "Органолептика":  "TP",   # вкус, запах
        "Упаковка":       "TP",
        "Вместимость":    "FP",   # keywords мусор: "местных тут", "беру", "нее"
        "Цена":           "TP",
        "Логистика":      "TP",
        "Свежесть":       "TP",   # сроки годности — важно для еды
        "Запах":          "TP",   # дубль органолептики но релевантен
        # residual medoid-named
        "зелёную гречку":   "FP",   # название разновидности, не аспект качества
        "гречка":           "FP",   # название продукта
        "увелки":           "FP",   # бренд
        "продукт":          "FP",   # мета
        "круп":             "FP",   # категория
        "горьковатый вкус": "TP",   # вкусовой аспект (дубль, но валидный)
        "2500":             "FP",   # число
        "годности хорошие": "TP",   # срок годности (дубль свежести)
        "проблем":          "FP",   # слишком размыто
        "пупырку":          "FP",   # мусор
    },

    # ── Карты Таро ─────────────────────────────────────────────────────
    54581151: {
        # anchor-named
        "Удобство":          "TP",   # удобство тасования/использования карт
        "Качество":          "TP",
        "Органолептика":     "FP",   # запах/вкус нерелевантны для карт
        "Внешний вид":       "TP",
        "Упаковка":          "TP",
        "Комфорт":           "TP",   # тактильные ощущения карт — обсуждается
        "Продавец":          "TP",   # продавец/издательство
        "Состояние":         "TP",   # дефекты при получении
        "Цена":              "TP",
        "Вместимость":       "FP",   # мусор: "пространства", "уголок"
        "Функциональность":  "TP",   # "энергия", "работа" колоды — эзотерика, но обсуждается
        "Логистика":         "TP",
        "Соответствие":      "TP",   # размер, оригинал vs реплика
        "Содержание":        "TP",   # сюжеты, тематика, гайдбук
        "Функционал":        "FP",   # дубль Функциональности, пустой
        "Запах":             "FP",   # нерелевантен для карт
        # residual medoid-named
        "формата":              "TP",   # формат/версия колоды
        "хорошие цвета":        "TP",   # цветопередача
        "плотные карты":        "TP",   # плотность карт
        "покрытие":             "TP",   # ламинация/глянец
        "коробочка":            "TP",   # упаковка (дубль)
        "рук":                  "FP",   # мусор
        "сотрудникам":          "FP",   # мусор
        "толстые":              "TP",   # толщина карт
        "вмчтенка":             "FP",   # непонятная аббревиатура
        "дефект":               "TP",   # дефекты — валидный аспект
        "ламинации":            "TP",   # качество ламинации
        "картинки":             "TP",   # арт, рисунки
        "приятная колода":      "FP",   # общее впечатление
        "товар":                "FP",   # мета
        "ощущениям":            "FP",   # слишком размыто
        "толкование":           "TP",   # толкование/гайдбук
        "колода замечательная": "FP",   # общее впечатление
        "эта колода":           "FP",   # мета
        "печать":               "TP",   # качество печати
    },

    # ── Средство от тараканов ──────────────────────────────────────────
    619500952: {
        # anchor-named
        "Качество":          "TP",   # общее качество средства
        "Удобство":          "TP",   # удобство применения
        "Состав":            "FP",   # keywords "отдельные особи" — мусор
        "Упаковка":          "TP",
        "Комфорт":           "FP",   # "большую поверхность" — не про комфорт
        "Здоровье":          "TP",   # безопасность для людей
        "Функциональность":  "TP",   # эффективность средства
        "Содержание":        "FP",   # "свою историю" — нерелевантно
        "Продавец":          "TP",   # продавец
        "Вместимость":       "FP",   # "места", "этаже" — локация, не аспект
        "Органолептика":     "TP",   # запах средства — обсуждается активно
        "Состояние":         "FP",   # "хулиганов", "трупов" — про тараканов, не про товар
        "Цена":              "TP",
        "Логистика":         "TP",
        "Поедаемость":       "FP",   # абсурд для средства от тараканов
        "Свежесть":          "FP",   # нерелевантно
        "Запах":             "TP",   # запах — один из главных аспектов
        "Функционал":        "FP",   # дубль Функциональности
        # residual medoid-named
        "пор тараканов":     "FP",   # мета
        "отличное средство": "FP",   # общее впечатление
        "обработки":         "TP",   # процесс обработки — валидный аспект
        "питомцев":          "TP",   # безопасность для животных
        "ловушками":         "FP",   # про другие средства
        "засилье":           "FP",   # мусор
        "ванной":            "FP",   # локация
        "углы":              "FP",   # локация
        "перестраховки":     "TP",   # защита/безопасность при применении
        "столешницу":        "FP",   # локация
        "эффекте":           "TP",   # эффективность
        "квартиру":          "FP",   # локация
        "рыжих насекомых":   "FP",   # мета про тараканов
        "применения":        "TP",   # процесс применения
        "два-три":           "FP",   # таймфрейм, не аспект
    },
}


# =========================================================================
# 3. РАСЧЁТ МЕТРИК
# =========================================================================

def load_true_labels(csv_path, product_ids):
    """Загрузить true_labels из CSV для нужных товаров."""
    reviews = defaultdict(dict)  # {pid: {review_id: {aspect: score}}}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["nm_id"])
            if pid not in product_ids:
                continue
            rid = row["id"]
            raw = row.get("true_labels", "")
            if not raw or raw == "{}":
                continue
            try:
                labels = json.loads(raw.replace('""', '"'))
            except:
                labels = json.loads(raw)
            reviews[pid][rid] = labels
    return reviews


def compute_product_mae(true_labels, pred_per_review, mapping):
    """
    Product-level MAE: для каждого (товар, true_aspect) считаем
    |mean(true) - mean(pred)|, где pred собран через manual mapping.
    """
    # Собираем pred scores по true aspect через mapping
    # reverse mapping: true_aspect → [pred_aspect1, pred_aspect2, ...]
    reverse_map = defaultdict(list)
    for pred_a, true_a in mapping.items():
        if true_a is not None:
            reverse_map[true_a].append(pred_a)

    results = []
    for true_aspect, pred_aspects in reverse_map.items():
        true_scores = []
        pred_scores = []

        for rid, labels in true_labels.items():
            if true_aspect in labels:
                true_scores.append(labels[true_aspect])

            # Собираем pred scores из всех замапленных аспектов
            rid_preds = []
            if rid in pred_per_review:
                for pa in pred_aspects:
                    if pa in pred_per_review[rid]:
                        rid_preds.append(pred_per_review[rid][pa])
            if rid_preds:
                pred_scores.append(np.mean(rid_preds))

        n_true = len(true_scores)
        n_pred = len(pred_scores)

        if n_true >= 3 and n_pred > 0:
            mean_true = np.mean(true_scores)
            mean_pred = np.mean(pred_scores)
            mae = abs(mean_true - mean_pred)
            results.append({
                "aspect": true_aspect,
                "true_mean": mean_true,
                "pred_mean": mean_pred,
                "mae": mae,
                "n_true": n_true,
                "n_pred": n_pred,
            })

    return results


def compute_recall(true_labels, mapping):
    """
    Aspect-level recall: какую долю true аспектов покрывает mapping?
    """
    # Все уникальные true аспекты
    all_true = set()
    for rid, labels in true_labels.items():
        all_true.update(labels.keys())

    # Покрытые: есть хотя бы один pred → true
    covered = set(v for v in mapping.values() if v is not None)

    return {
        "total_true": len(all_true),
        "covered": len(all_true & covered),
        "missed": sorted(all_true - covered),
        "recall": len(all_true & covered) / len(all_true) if all_true else 0,
    }


def compute_mention_recall(true_labels, pred_per_review, mapping):
    """
    Mention recall (per-review): для каждого (review, true_aspect),
    есть ли хотя бы одна предсказанная оценка?
    """
    reverse_map = defaultdict(list)
    for pred_a, true_a in mapping.items():
        if true_a is not None:
            reverse_map[true_a].append(pred_a)

    total_mentions = 0
    covered_mentions = 0

    for rid, labels in true_labels.items():
        for true_aspect in labels:
            total_mentions += 1
            if true_aspect in reverse_map and rid in pred_per_review:
                for pa in reverse_map[true_aspect]:
                    if pa in pred_per_review[rid]:
                        covered_mentions += 1
                        break

    return {
        "total": total_mentions,
        "covered": covered_mentions,
        "recall": covered_mentions / total_mentions if total_mentions else 0,
    }


def compute_precision(precision_dict):
    """Считает precision из TP/FP разметки."""
    anchor_tp = anchor_fp = 0
    resid_tp = resid_fp = 0

    for aspect, verdict in precision_dict.items():
        # Определяем anchor vs residual: anchor-named аспекты
        # начинаются с большой буквы (кириллица)
        is_anchor = aspect[0].isupper() if aspect else False

        if verdict == "TP":
            if is_anchor:
                anchor_tp += 1
            else:
                resid_tp += 1
        elif verdict == "FP":
            if is_anchor:
                anchor_fp += 1
            else:
                resid_fp += 1

    total_tp = anchor_tp + resid_tp
    total_fp = anchor_fp + resid_fp
    total = total_tp + total_fp

    return {
        "anchor_tp": anchor_tp,
        "anchor_fp": anchor_fp,
        "anchor_total": anchor_tp + anchor_fp,
        "anchor_precision": anchor_tp / (anchor_tp + anchor_fp) if (anchor_tp + anchor_fp) else 0,
        "residual_tp": resid_tp,
        "residual_fp": resid_fp,
        "residual_total": resid_tp + resid_fp,
        "residual_precision": resid_tp / (resid_tp + resid_fp) if (resid_tp + resid_fp) else 0,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_precision": total_tp / total if total else 0,
    }


# =========================================================================
# 4. MAIN
# =========================================================================

PRODUCT_NAMES = {
    15430704:  "Гречка",
    54581151:  "Карты Таро",
    619500952: "Ср-во от тараканов",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-review", required=True, help="suite_eval_per_review.json")
    parser.add_argument("--csv", required=True, help="merged_checked_reviews.csv")
    args = parser.parse_args()

    # Загрузка данных
    product_ids = set(MANUAL_MAPPING.keys())

    with open(args.per_review, encoding="utf-8") as f:
        pred_all = json.load(f)

    true_all = load_true_labels(args.csv, product_ids)

    # ── Precision ──────────────────────────────────────────────────
    print("=" * 70)
    print("MANUAL PRECISION")
    print("=" * 70)

    all_prec = {}
    for pid in sorted(product_ids):
        name = PRODUCT_NAMES[pid]
        prec = compute_precision(MANUAL_PRECISION[pid])
        all_prec[pid] = prec
        print(f"\n  {name} ({pid}):")
        print(f"    Anchor:   {prec['anchor_tp']} TP / {prec['anchor_fp']} FP"
              f"  ->  Precision = {prec['anchor_precision']:.3f}"
              f"  ({prec['anchor_total']} aspects)")
        print(f"    Residual: {prec['residual_tp']} TP / {prec['residual_fp']} FP"
              f"  ->  Precision = {prec['residual_precision']:.3f}"
              f"  ({prec['residual_total']} aspects)")
        print(f"    TOTAL:    {prec['total_tp']} TP / {prec['total_fp']} FP"
              f"  ->  Precision = {prec['total_precision']:.3f}"
              f"  ({prec['total_tp'] + prec['total_fp']} aspects)")

    # Macro precision
    anchor_precs = [all_prec[pid]["anchor_precision"] for pid in sorted(product_ids)]
    total_precs = [all_prec[pid]["total_precision"] for pid in sorted(product_ids)]
    print(f"\n  -- Macro Precision (anchor-only): {np.mean(anchor_precs):.3f}")
    print(f"  -- Macro Precision (all aspects):  {np.mean(total_precs):.3f}")

    # ── Recall ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ASPECT RECALL (manual mapping)")
    print("=" * 70)

    all_recalls = []
    for pid in sorted(product_ids):
        name = PRODUCT_NAMES[pid]
        rec = compute_recall(true_all[pid], MANUAL_MAPPING[pid])
        all_recalls.append(rec["recall"])
        print(f"\n  {name} ({pid}):")
        print(f"    True aspects: {rec['total_true']}, Covered: {rec['covered']}"
              f"  ->  Recall = {rec['recall']:.3f}")
        if rec["missed"]:
            print(f"    Missed: {rec['missed']}")

    print(f"\n  -- Macro Recall: {np.mean(all_recalls):.3f}")

    # ── Mention Recall ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("MENTION RECALL (per-review, manual mapping)")
    print("=" * 70)

    all_mention = []
    for pid in sorted(product_ids):
        name = PRODUCT_NAMES[pid]
        spid = str(pid)
        pred_reviews = pred_all.get(spid, {})
        mr = compute_mention_recall(true_all[pid], pred_reviews, MANUAL_MAPPING[pid])
        all_mention.append(mr["recall"])
        print(f"\n  {name} ({pid}):")
        print(f"    Mentions: {mr['total']}, Covered: {mr['covered']}"
              f"  ->  Recall = {mr['recall']:.3f}")

    print(f"\n  -- Macro Mention Recall: {np.mean(all_mention):.3f}")

    # ── Product MAE ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRODUCT-LEVEL MAE (n_true >= 3, manual mapping)")
    print("=" * 70)

    all_mae_pairs = []
    per_product_mae = []

    for pid in sorted(product_ids):
        name = PRODUCT_NAMES[pid]
        spid = str(pid)
        pred_reviews = pred_all.get(spid, {})
        results = compute_product_mae(true_all[pid], pred_reviews, MANUAL_MAPPING[pid])

        print(f"\n  {name} ({pid}):")
        print(f"    {'Aspect':<25s} {'True':>6s} {'Pred':>6s} {'d':>6s}  {'n_true':>6s} {'n_pred':>6s}")
        print(f"    {'-'*65}")

        product_maes = []
        for r in sorted(results, key=lambda x: x["mae"]):
            print(f"    {r['aspect']:<25s} {r['true_mean']:>6.2f} {r['pred_mean']:>6.2f}"
                  f" {r['mae']:>6.2f}  {r['n_true']:>6d} {r['n_pred']:>6d}")
            product_maes.append(r["mae"])
            all_mae_pairs.append(r["mae"])

        if product_maes:
            pmae = np.mean(product_maes)
            per_product_mae.append(pmae)
            print(f"    {'':>25s} MAE = {pmae:.3f}  (n_pairs={len(product_maes)})")

    print(f"\n  -- Global Product MAE (n>=3): {np.mean(all_mae_pairs):.3f}"
          f"  (n={len(all_mae_pairs)} pairs)")
    print(f"  -- Macro Product MAE:        {np.mean(per_product_mae):.3f}")

    # ── Итоговая сводка ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ИТОГОВАЯ СВОДКА (held-out, manual mapping + manual precision)")
    print("=" * 70)
    print(f"  Macro Precision (anchor):     {np.mean(anchor_precs):.3f}")
    print(f"  Macro Precision (all):        {np.mean(total_precs):.3f}")
    print(f"  Macro Recall (aspect):        {np.mean(all_recalls):.3f}")
    print(f"  Macro Mention Recall:         {np.mean(all_mention):.3f}")
    print(f"  Product MAE (n>=3, global):    {np.mean(all_mae_pairs):.3f}")
    print(f"  Product MAE (n>=3, macro):     {np.mean(per_product_mae):.3f}")
    print()


if __name__ == "__main__":
    main()
