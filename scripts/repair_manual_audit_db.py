from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_recalc.storage import (
    connect,
    now_iso,
    save_app_meta,
    upsert_gold_decision,
    upsert_review_status,
    upsert_system_decision,
)
from scripts.recompute_manual_audit_metrics import DEFAULT_DATASET_PATH, DEFAULT_DB_PATH, DEFAULT_RUN_DIR, build_audit_base


REPAIR_PLAN: dict[str, dict[str, dict[str, str] | list[str]]] = {
    "0SjoNYwBI2GFApLnYLFT": {"tp": {"04982fbccbb20085": "Внешний вид"}},
    "0kGnbocBxpMXtsmO6b-X": {"tp": {"f1bc25e638d26275": "Внешний вид", "237383e5e5c957fd": "Качество"}},
    "0n21jokBKNxVyjcCokFA": {"tp": {"7a911a9f1a1932a9": "Внешний вид"}},
    "1cqzH4gBuWYO9ah29ZSm": {
        "tp": {
            "ba78c69da0104bff": "Комплектация",
            "3291a8acdbd17a6f": "Инструкция",
            "f8503b15c8277a13": "Качество",
            "2c8dc9c932f1db18": "Упаковка",
        }
    },
    "or2HP4UBk9xzjhYZhQTZ": {"tp": {"44c8fe6387c75824": "Качество", "56879fb5d6c38a10": "Упаковка"}},
    "pOdO24kBqdFzcemq94cg": {"tp": {"90770fee8d64adf5": "Качество"}},
    "pPWH3rCMTh8UJe88yf76": {
        "tp": {"e5cb3067bbe7c329": "Эффективность", "a6376c39fd62b9d6": "Запах"},
        "duplicate": {"65d71e71462c5f6f": "Эффективность"},
    },
    "pkXkyGAiqgz1BSTl64UN": {
        "tp": {"20b813dc3358362d": "Эффективность"},
        "duplicate": {
            "5d9b9496454a415c": "Эффективность",
            "770fa7c1cb3cf74b": "Эффективность",
        },
    },
    "pmkYV5RRimfkWfGkxOSR": {
        "tp": {"9546e0999aa94746": "Эффективность"},
        "duplicate": {"102ba36f8f4c6a8e": "Эффективность"},
    },
    "pst3GYsBo8kheRYZnRUQ": {"tp": {"47cdb13abafdb169": "Качество"}},
    "qISYvIkBuX2t2Z2-wcOq": {"tp": {"ed3e44ea7304710e": "Качество"}},
    "qtGmlBZoqHpCxD51FWK8": {"tp": {"72d00ca91b328317": "Эффективность"}},
    "qvlK59YfX0MgFLbM983p": {
        "tp": {"8eec7c2e63ffb06f": "Эффективность", "da2d19b4bfaa7566": "Запах"},
        "duplicate": {"376ba942e5a792e3": "Эффективность"},
    },
    "qzLgmKKIJiostERjCNlB": {"tp": {"cc073b7636bebe63": "Эффективность", "ae5c6360bf981ee9": "Безопасность"}},
    "rYBznfoMj5cdgynR4OFP": {
        "tp": {
            "07d3dec09931660a": "Безопасность",
            "782696081cfbee42": "Эффективность",
            "ef8f87551fb4c906": "Запах",
        }
    },
    "rkz06ocBqdFzcemqeliU": {"tp": {"6ac9feb21f51b447": "Качество"}},
    "ro5p27XgHa48n8VTpJcv": {"tp": {"2bcf25c62900f954": "Эффективность"}},
    "rx5nP40B_8RHzXwXslJe": {"tp": {"eee9143ebf80fc2f": "Качество"}},
    "sY3hy42OZPqxzS9ToOwm": {
        "tp": {"8db9a0c69a8cdfd7": "Эффективность"},
        "duplicate": {"26026499b6b9b1a9": "Эффективность"},
    },
    "seAjtIcBqd2abVcyhkCs": {"tp": {"4cd7ab0c6405e26a": "Упаковка"}},
    "sgimUlpx3cyguXl08gJB": {"tp": {"93955f1921c8ce8e": "Эффективность", "f7f3478d46d46f72": "Запах"}},
    "sojbJ4MHeJlattT0vyZj": {"tp": {"0fcd684a4c5154d3": "Эффективность"}},
    "tPUg0OE1yTiHUkO2pttQ": {"tp": {"bd7a31fe87b19fac": "Качество"}},
    "tfRkmHwcrBv1tfYTpdo0": {
        "tp": {"7be86acc5c7ab41c": "Эффективность"},
        "duplicate": {"0a6c71b2e073fc5d": "Эффективность"},
    },
    "tmdndYsBGgNwMUDv-Ghk": {"tp": {"f56290295150b332": "Качество"}},
    "u5GTP4UBtPujtG5sBzgZ": {"tp": {"23e0b517853050ac": "Качество"}},
    "uZ6JHIwBK9xQ7jEJkIOY": {"tp": {"d64af50cea9c11c7": "Качество"}},
    "uZcMuHf45JsVxU3hTB9r": {
        "tp": {"3c508db7ce756066": "Эффективность"},
        "duplicate": {"8d6b4c252da96f14": "Эффективность"},
    },
    "v9w1azMIwgcBcm0wtM5v": {
        "tp": {"f1f05615094739fd": "Эффективность", "79a3ded17e3c9f92": "Использование"},
        "duplicate": {"23e787d8eec589a6": "Эффективность"},
    },
    "vo12LocBk9xzjhYZpVbF": {"tp": {"ae04c3b99819b928": "Качество"}},
    "vwpW5lX7JlQWUCx6kVLH": {
        "tp": {"ac7d63d231633abb": "Эффективность"},
        "duplicate": {"18d94d0216f642f5": "Эффективность"},
    },
    "wQdcRedVtOBqNOasOv3h": {"tp": {"ef6ae3ae7f9fc0ab": "Эффективность"}},
    "waPBr6S10uSqaKrTMzTa": {
        "tp": {"55d2b28a19c365b6": "Эффективность"},
        "duplicate": {"1c463de6235d9c03": "Эффективность"},
    },
    "wyJaDjIXJuvkTOmOkbyt": {
        "tp": {"baf8bf63ff56cd3f": "Эффективность"},
        "duplicate": {"6e43aef611be1399": "Эффективность"},
    },
    "x1SHcIQBbunnhI3gJczX": {"tp": {"2f0295b421f55188": "Качество"}},
    "x1TGHYcBT-IFIyX_SQJo": {"tp": {"e619fc41f2170773": "Качество"}},
    "xRLyMbL078vsKD4Rbxsv": {
        "tp": {"8afca6e3a6715120": "Эффективность"},
        "duplicate": {
            "9191448cecd8b563": "Эффективность",
            "c945d92a8184e3e4": "Эффективность",
            "5a4e617acc3e045f": "Эффективность",
        },
    },
    "xllJpPtG1Nym14EOWGsp": {"tp": {"e58315649bafd570": "Эффективность"}},
    "yYtHNV8HQYAmT7xXWD5s": {
        "tp": {"f23da496caf1d5c2": "Эффективность", "cce498e6528e45a9": "Запах"},
        "duplicate": {"429d1b64bd45bb1e": "Эффективность"},
    },
    "yaILJIwBwX70cIf6QHvA": {
        "tp": {"670b78e8b23b6b01": "Качество", "b8dc3df83eba2a0b": "Цена"},
        "duplicate": {"7da16cf5f3e31a13": "Качество"},
    },
    "ybIanQydzNaJZaGSY1jS": {"tp": {"db5ffa1bb3366608": "Эффективность"}},
    "ym_c59de42a475ceefe_0001": {
        "tp": {"015400665285e0f5": "Заселение", "8a6e7ddc29ccb1c4": "Обслуживание"},
        "duplicate": {"e4d970b2a43ea99f": "Заселение"},
    },
    "z4RKPYcBMJfQQSHDKDQo": {
        "tp": {
            "cdf9caf4b9882036": "Качество",
            "c6d1a8ec86079267": "Упаковка",
            "80906c5edb60f1d7": "Цена",
        },
        "duplicate": {"b945a8cd79fc7652": "Качество"},
    },
    "z6mgSIsB5HYuAcv3otFh": {"tp": {"a74b5033c342abba": "Качество", "cf7b3cd5e6d85b10": "Упаковка"}},
    "z7PHKYsBwbGU35ygZ1Kc": {"tp": {"efb1453b4ffca6bb": "Цена"}},
    "zLzWS4YBxpMXtsmOijk7": {"tp": {"8f55e3105fe7a398": "Сервис", "308c3fee5351bcc8": "Упаковка"}},
    "zTTQd8XzUgUjtSX7Vz34": {
        "tp": {"6254c39d3a9b507e": "Эффективность"},
        "duplicate": {"13b53b94577d258d": "Эффективность"},
    },
    "zWyoH3VQ6xa4Ffxzsyrk": {"tp": {"46ef04205855db98": "Эффективность"}},
    "zcXraRrdTetqknB3oSCP": {"tp": {"56bf22bceb63d21c": "Качество"}},
}

SYSTEM_FIXES: dict[str, dict[str, str]] = {
    "0b767423b4eca312": {"manual_decision": "TP", "mapped_gold_aspect": "Размер"},
    "767a5d9bdc0be449": {"manual_decision": "DUPLICATE", "mapped_gold_aspect": "Качество"},
    "dc265eae7b48179f": {"manual_decision": "TP", "mapped_gold_aspect": "Качество"},
    "296b4890a87b12c4": {"manual_decision": "FP", "mapped_gold_aspect": ""},
    "a4030180cfce78be": {"manual_decision": "FP", "mapped_gold_aspect": ""},
    "a399a86b79537353": {"manual_decision": "FP", "mapped_gold_aspect": ""},
    "4279f75ce086e502": {"manual_decision": "FP", "mapped_gold_aspect": ""},
}


def _group_gold(dataset_gold) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in dataset_gold.itertuples(index=False):
        grouped.setdefault(str(row.review_id), []).append(
            {
                "gold_aspect": str(row.gold_aspect),
                "gold_rating": float(row.gold_rating),
            }
        )
    return grouped


def _group_expected_system(expected_system) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in expected_system.itertuples(index=False):
        grouped.setdefault(str(row.review_id), []).append(
            {
                "prediction_id": str(row.prediction_id),
                "review_id": str(row.review_id),
                "system_aspect": str(row.system_aspect),
                "system_rating": float(row.system_rating),
                "aspect_source": str(row.aspect_source),
            }
        )
    return grouped


def _upsert_system_rows(conn, review_id: str, system_rows: list[dict[str, Any]], plan: dict[str, dict[str, str] | list[str]]) -> dict[str, str]:
    tp_map = dict(plan.get("tp", {}))
    duplicate_map = dict(plan.get("duplicate", {}))
    unclear_map = dict(plan.get("unclear", {}))
    matched_tp_by_gold: dict[str, str] = {}

    for row in system_rows:
        prediction_id = row["prediction_id"]
        decision = "FP"
        mapped_gold_aspect = ""
        if prediction_id in tp_map:
            decision = "TP"
            mapped_gold_aspect = tp_map[prediction_id]
            matched_tp_by_gold.setdefault(mapped_gold_aspect, prediction_id)
        elif prediction_id in duplicate_map:
            decision = "DUPLICATE"
            mapped_gold_aspect = duplicate_map[prediction_id]
        elif prediction_id in unclear_map:
            decision = "UNCLEAR"
            mapped_gold_aspect = unclear_map[prediction_id]

        upsert_system_decision(
            conn,
            {
                "prediction_id": prediction_id,
                "review_id": review_id,
                "system_aspect": row["system_aspect"],
                "system_rating": row["system_rating"],
                "aspect_source": row["aspect_source"],
                "manual_decision": decision,
                "mapped_gold_aspect": mapped_gold_aspect,
                "manual_sentiment_decision": "",
                "comment": "",
                "source": "human",
                "confirmed_by_human": True,
                "committed": True,
                "updated_at": now_iso(),
            },
        )
    return matched_tp_by_gold


def _upsert_gold_rows(conn, review_id: str, gold_rows: list[dict[str, Any]], matched_tp_by_gold: dict[str, str], plan: dict[str, dict[str, str] | list[str]]) -> None:
    gold_unclear = set(plan.get("gold_unclear", []))
    for row in gold_rows:
        gold_aspect = row["gold_aspect"]
        if gold_aspect in matched_tp_by_gold:
            status = "FOUND"
            matched_prediction_id = matched_tp_by_gold[gold_aspect]
        elif gold_aspect in gold_unclear:
            status = "UNCLEAR"
            matched_prediction_id = ""
        else:
            status = "FN"
            matched_prediction_id = ""

        upsert_gold_decision(
            conn,
            {
                "review_id": review_id,
                "gold_aspect": gold_aspect,
                "gold_rating": row["gold_rating"],
                "status": status,
                "matched_system_prediction_id": matched_prediction_id,
                "comment": "",
                "committed": True,
                "updated_at": now_iso(),
            },
        )


def _repair_missing_reviews(conn, expected_by_review: dict[str, list[dict[str, Any]]], gold_by_review: dict[str, list[dict[str, Any]]]) -> None:
    for review_id, plan in REPAIR_PLAN.items():
        system_rows = expected_by_review[review_id]
        gold_rows = gold_by_review[review_id]
        matched_tp_by_gold = _upsert_system_rows(conn, review_id, system_rows, plan)
        _upsert_gold_rows(conn, review_id, gold_rows, matched_tp_by_gold, plan)
        upsert_review_status(conn, review_id, "repair_batch", "done", committed=True)


def _repair_blank_system_statuses(conn, expected_system) -> None:
    expected_lookup = {
        str(row.prediction_id): {
            "prediction_id": str(row.prediction_id),
            "review_id": str(row.review_id),
            "system_aspect": str(row.system_aspect),
            "system_rating": float(row.system_rating),
            "aspect_source": str(row.aspect_source),
        }
        for row in expected_system.itertuples(index=False)
    }
    for prediction_id, fix in SYSTEM_FIXES.items():
        row = expected_lookup[prediction_id]
        upsert_system_decision(
            conn,
            {
                "prediction_id": prediction_id,
                "review_id": row["review_id"],
                "system_aspect": row["system_aspect"],
                "system_rating": row["system_rating"],
                "aspect_source": row["aspect_source"],
                "manual_decision": fix["manual_decision"],
                "mapped_gold_aspect": fix["mapped_gold_aspect"],
                "manual_sentiment_decision": "",
                "comment": "",
                "source": "human",
                "confirmed_by_human": True,
                "committed": True,
                "updated_at": now_iso(),
            },
        )


def _resolve_found_without_match(conn) -> tuple[int, int]:
    gold_rows = conn.execute(
        """
        SELECT review_id, gold_aspect
        FROM gold_decisions
        WHERE status = 'FOUND'
          AND COALESCE(NULLIF(TRIM(matched_system_prediction_id), ''), '') = ''
        """
    ).fetchall()
    fixed = 0
    downgraded = 0
    for gold_row in gold_rows:
        review_id = str(gold_row["review_id"])
        gold_aspect = str(gold_row["gold_aspect"])
        tp_rows = conn.execute(
            """
            SELECT prediction_id
            FROM system_decisions
            WHERE review_id = ?
              AND manual_decision = 'TP'
              AND COALESCE(NULLIF(TRIM(mapped_gold_aspect), ''), '') = ?
            ORDER BY prediction_id
            """,
            (review_id, gold_aspect),
        ).fetchall()
        if len(tp_rows) == 1:
            conn.execute(
                """
                UPDATE gold_decisions
                SET matched_system_prediction_id = ?, committed = 1, updated_at = ?
                WHERE review_id = ? AND gold_aspect = ?
                """,
                (str(tp_rows[0]["prediction_id"]), now_iso(), review_id, gold_aspect),
            )
            fixed += 1
            continue

        has_any_map = conn.execute(
            """
            SELECT COUNT(*)
            FROM system_decisions
            WHERE review_id = ?
              AND COALESCE(NULLIF(TRIM(mapped_gold_aspect), ''), '') = ?
            """,
            (review_id, gold_aspect),
        ).fetchone()[0]
        fallback_status = "UNCLEAR" if has_any_map else "FN"
        conn.execute(
            """
            UPDATE gold_decisions
            SET status = ?, matched_system_prediction_id = '', committed = 1, updated_at = ?
            WHERE review_id = ? AND gold_aspect = ?
            """,
            (fallback_status, now_iso(), review_id, gold_aspect),
        )
        downgraded += 1
    conn.commit()
    return fixed, downgraded


def _normalize_done_review_status(conn) -> int:
    rows = conn.execute("SELECT review_id, status FROM review_status").fetchall()
    fixed = 0
    for row in rows:
        review_id = str(row["review_id"])
        status = str(row["status"])
        if status != "done":
            continue
        system_pending = conn.execute(
            "SELECT COUNT(*) FROM system_decisions WHERE review_id = ? AND committed = 0",
            (review_id,),
        ).fetchone()[0]
        gold_pending = conn.execute(
            "SELECT COUNT(*) FROM gold_decisions WHERE review_id = ? AND committed = 0",
            (review_id,),
        ).fetchone()[0]
        committed_value = 1 if system_pending == 0 and gold_pending == 0 else 0
        before = conn.execute("SELECT committed FROM review_status WHERE review_id = ?", (review_id,)).fetchone()[0]
        if int(before) == committed_value:
            continue
        conn.execute(
            "UPDATE review_status SET committed = ?, updated_at = ? WHERE review_id = ?",
            (committed_value, now_iso(), review_id),
        )
        fixed += 1
    conn.commit()
    return fixed


def run(db_path: Path, run_dir: Path, dataset_path: Path) -> dict[str, int]:
    if set(REPAIR_PLAN) != set(REPAIR_PLAN.keys()):
        raise AssertionError("repair review ids must be unique")

    base = build_audit_base(db_path=db_path, run_dir=run_dir, dataset_path=dataset_path)
    expected_by_review = _group_expected_system(base["expected_system"])
    gold_by_review = _group_gold(base["dataset_gold"])

    missing_from_expected = sorted(set(REPAIR_PLAN) - set(expected_by_review))
    missing_from_gold = sorted(set(REPAIR_PLAN) - set(gold_by_review))
    if missing_from_expected or missing_from_gold:
        raise RuntimeError(f"repair review ids missing in source data: expected={missing_from_expected}, gold={missing_from_gold}")

    conn = connect(db_path)
    try:
        _repair_missing_reviews(conn, expected_by_review, gold_by_review)
        _repair_blank_system_statuses(conn, base["expected_system"])
        found_fixed, found_downgraded = _resolve_found_without_match(conn)
        normalized_status = _normalize_done_review_status(conn)
        save_app_meta(conn, "repair_batch_review_ids", sorted(REPAIR_PLAN))
    finally:
        conn.close()

    return {
        "repair_reviews": len(REPAIR_PLAN),
        "blank_status_fixes": len(SYSTEM_FIXES),
        "found_fixed": found_fixed,
        "found_downgraded": found_downgraded,
        "normalized_done_status": normalized_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    summary = run(args.db_path, args.run_dir, args.dataset_path)
    for key, value in summary.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
