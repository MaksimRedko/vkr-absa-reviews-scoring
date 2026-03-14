import re
from typing import List, Dict, Set
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsMorphTagger, Doc
)


class NLPProcessor:
    """
    Модуль лингвистической предобработки отзывов.
    Отвечает за очистку текста, разбиение на предложения и извлечение
    семантически значимых лемм (кандидатов в аспекты).
    """

    def __init__(self, min_word_length: int = 3):
        print("⏳ Инициализация NLPProcessor (загрузка моделей Natasha)...")
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.min_word_length = min_word_length

        # Стоп-слова: мусор, который часто проходит POS-теггер, но не несет смысла для товаров
        self.stop_words: Set[str] = {
            'штука', 'раз', 'день', 'год', 'вчера', 'сегодня', 'завтра',
            'неделя', 'месяц', 'человек', 'очень', 'просто', 'вообще', 'свой'
        }
        print("✅ NLPProcessor готов.")

    def _clean_text(self, text: str) -> str:
        """Базовая очистка: фикс пробелов после пунктуации, удаление лишних символов."""
        if not text or not isinstance(text, str):
            return ""
        # Добавляем пробел после знаков препинания, если его там нет (пр: "хорошо.быстро" -> "хорошо. быстро")
        text = re.sub(r'(?<=[.,!?;:])(?=[^\s])', r' ', text)
        # Убираем все, кроме букв, цифр и базовой пунктуации (удаляет эмодзи)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def process_review(self, raw_text: str) -> Dict[str, List[str]]:
        """
        Обрабатывает один отзыв.
        Возвращает чистые предложения (для NLI) и леммы (для кластеризации).
        """
        clean_txt = self._clean_text(raw_text)
        if not clean_txt:
            return {"sentences": [], "lemmas": []}

        doc = Doc(clean_txt)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        sentences = [sent.text for sent in doc.sents if len(sent.text.split()) > 1]
        valid_lemmas = []

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
            lemma = token.lemma.lower()

            # Оставляем только Существительные, Прилагательные и Глаголы
            if token.pos not in ['NOUN', 'ADJ', 'VERB']:
                continue

            # Фильтр по длине и стоп-словам
            if len(lemma) < self.min_word_length or lemma in self.stop_words:
                continue

            valid_lemmas.append(lemma)

        return {
            "sentences": sentences,
            "lemmas": valid_lemmas
        }

    def extract_global_candidates(self, corpus: List[str], max_candidates: int = 150) -> List[str]:
        """
        Анализирует ВЕСЬ корпус отзывов на один товар и возвращает топ статистически
        значимых слов-кандидатов с использованием TF-IDF.

        Args:
            corpus: Список всех отзывов на конкретный товар.
            max_candidates: Сколько слов передать дальше в алгоритм кластеризации.
        """
        print(f"🔍 Извлечение кандидатов из {len(corpus)} отзывов...")

        # 1. Извлекаем леммы из всех документов
        docs_lemmas = []
        for text in corpus:
            res = self.process_review(text)
            if res["lemmas"]:
                # Склеиваем леммы обратно в строку для TF-IDF векторайзера
                docs_lemmas.append(" ".join(res["lemmas"]))

        if not docs_lemmas:
            return []

        # 2. Применяем TF-IDF для фильтрации
        # min_df=0.01: слово должно встретиться хотя бы в 1% отзывов (убивает опечатки)
        # max_df=0.80: слово не должно быть в >80% отзывов (убивает слишком общие слова типа "товар")
        vectorizer = TfidfVectorizer(
            min_df=max(2, int(len(docs_lemmas) * 0.01)),  # минимум 2 документа
            max_df=0.80,
            max_features=max_candidates
        )

        try:
            vectorizer.fit_transform(docs_lemmas)
            candidates = vectorizer.get_feature_names_out().tolist()
            return candidates
        except ValueError:
            # Если данных слишком мало для TF-IDF, возвращаем просто самые частые слова
            print("⚠️ Слишком мало данных для TF-IDF, используем fallback частотный словарь.")
            all_words = " ".join(docs_lemmas).split()
            most_common = [word for word, count in Counter(all_words).most_common(max_candidates)]
            return most_common


# --- ТЕСТОВЫЙ ЗАПУСК ---
if __name__ == "__main__":
    # Имитация того, что ты передаешь готовый корпус из своей БД
    sample_corpus = [
        "Достоинства: Отлично очищают экран, не оставляют разводов. Недостатки: Нет. Комментарий: Буду брать еще.",
        "Достоинства: Салфетки как салфетки. Недостатки: Быстро высыхают в тубе. Комментарий: Хватило на месяц.",
        "Достоинства: Цена. Недостатки: Оставляют жуткие разводы на матрице ноутбука! Комментарий: Испортил покрытие.",
        "Достоинства: Влажные, большая туба. Комментарий: Беру 2 раз, протираю телевизор и телефон.",
        "Комментарий: Приехали сухие! Просто кусок сухой бумаги, требую возврат.",
        "",  # Имитация пустой строки (строка 1 из твоей БД)
        "Достоинства: 👍"  # Имитация эмодзи (строка 5 из твоей БД)
    ]

    processor = NLPProcessor()

    # 1. Тест извлечения глобальных кандидатов (Пойдет в Кластеризацию)
    candidates = processor.extract_global_candidates(sample_corpus, max_candidates=20)
    print(f"\n🎯 Топ кандидатов для кластеризации ({len(candidates)} шт):")
    print(candidates)

    # 2. Тест обработки одного отзыва (Пойдет в Сентимент-анализ)
    print("\n📝 Разбор конкретного отзыва:")
    res = processor.process_review(sample_corpus[0])
    print(f"Предложения: {res['sentences']}")
    print(f"Леммы: {res['lemmas']}")