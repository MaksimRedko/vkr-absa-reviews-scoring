import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


RUBERT_ENCODER_MODEL = "cointegrated/rubert-tiny2"
RUBERT_NLI_MODEL = "cointegrated/rubert-base-cased-nli-threeway"
QWEN_NAMER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_encoder(cache_root: Path) -> None:
    print(f"[encoder] Скачиваю/проверяю {RUBERT_ENCODER_MODEL} (cache: {cache_root}) ...")
    ensure_dir(cache_root)
    SentenceTransformer(RUBERT_ENCODER_MODEL, cache_folder=str(cache_root))
    print("[encoder] Готово.")


def download_nli(cache_root: Path) -> None:
    nli_dir = cache_root / "rubert-nli"
    print(f"[nli] Скачиваю/проверяю {RUBERT_NLI_MODEL} (cache: {nli_dir}) ...")
    ensure_dir(nli_dir)
    AutoTokenizer.from_pretrained(RUBERT_NLI_MODEL, cache_dir=str(nli_dir))
    AutoModel.from_pretrained(RUBERT_NLI_MODEL, cache_dir=str(nli_dir))
    print("[nli] Готово.")


def download_qwen_namer(cache_root: Path) -> None:
    qwen_dir = cache_root / "qwen2_5_1_5b_instruct"
    print(f"[qwen] Скачиваю/проверяю {QWEN_NAMER_MODEL} (cache: {qwen_dir}) ...")
    ensure_dir(qwen_dir)
    AutoTokenizer.from_pretrained(QWEN_NAMER_MODEL, cache_dir=str(qwen_dir))
    AutoModelForCausalLM.from_pretrained(QWEN_NAMER_MODEL, cache_dir=str(qwen_dir))
    print("[qwen] Готово.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Скачать локально модели для ABSA пайплайна."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Каталог для кеша моделей (по умолчанию ./models).",
    )
    args = parser.parse_args()

    cache_root = Path(args.models_dir).resolve()
    ensure_dir(cache_root)

    print(f"Использую каталог моделей: {cache_root}")

    download_encoder(cache_root)
    download_nli(cache_root)
    download_qwen_namer(cache_root)

    print("Все модели скачаны. Повторный запуск ничего не перекачивает.")


if __name__ == "__main__":
    main()

