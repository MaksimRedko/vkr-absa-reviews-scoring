"""
Однократный экспорт NLI (rubert-base-cased-nli-threeway) в ONNX + динамическое INT8.

Запуск из корня репозитория:
  python scripts/export_nli_onnx.py

Пути по умолчанию рядом с HF-снапшотом из configs.configs (models.nli_path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from configs.configs import config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32-out",
        type=Path,
        default=_ROOT / "models" / "rubert-nli" / "nli_threeway_fp32.onnx",
        help="Куда сохранить float32 ONNX.",
    )
    parser.add_argument(
        "--int8-out",
        type=Path,
        default=_ROOT / "models" / "rubert-nli" / "nli_threeway_int8.onnx",
        help="Куда сохранить динамически квантованный INT8 ONNX.",
    )
    args = parser.parse_args()

    model_id = config.models.nli_path
    print(f"Загрузка модели из {model_id!r} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, local_files_only=True
    )
    model.eval()

    dummy = tokenizer(
        "Текст премисы",
        "Гипотеза про аспект",
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    args.fp32_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Экспорт FP32 ONNX → {args.fp32_out}")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(args.fp32_out),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )

    print(f"Динамическое квантование INT8 → {args.int8_out}")
    quantize_dynamic(
        model_input=str(args.fp32_out),
        model_output=str(args.int8_out),
        weight_type=QuantType.QInt8,
    )
    print("Готово. В configs.models.nli_onnx_quantized_path укажите путь к int8 ONNX.")


if __name__ == "__main__":
    main()
