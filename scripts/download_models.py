import argparse
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


RUBERT_ENCODER_MODEL = "cointegrated/rubert-tiny2"
RUBERT_NLI_MODEL = "cointegrated/rubert-base-cased-nli-twoway"


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

    print("Все модели скачаны. Повторный запуск ничего не перекачивает.")


if __name__ == "__main__":
    main()

