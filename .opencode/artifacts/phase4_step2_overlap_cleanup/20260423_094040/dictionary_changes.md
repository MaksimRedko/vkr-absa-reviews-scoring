# Dictionary Changes

## Synonyms removed from general anchors
- `Доставка`: removed `пришло`, `пришла`
- `Соответствие ожиданиям`: removed `соответствие`
- `Ассортимент`: removed `комплект`, `комплектация`
- `Общее впечатление`: removed `общее`, `хорошо`, `плохо`, `отлично`, `ужасно`

## Synonyms removed from domain anchors
- `physical_goods / Инструкция`: removed `схема`, `manual`
- `physical_goods / Удобство использования`: removed `эксплуатация`
- `physical_goods / Уход и обслуживание`: removed `обслуживать`
- `physical_goods / Крепления и соединения`: removed `соединение`
- `services / Запись на услугу`: removed `расписание`
- `services / Поддержка и сопровождение`: removed `сопровождение`
- `consumables / Норма кормления`: removed `расход`
- `hospitality / Номер`: removed `комната`
- `hospitality / Завтрак и питание`: removed `еда`
- `hospitality / Wi-Fi и интернет`: removed `сеть`, `сигнал`

## Duplicate / overlap decisions
- `services / Ожидание` against general `Загруженность`: treated as domain-preferred conflict in routing.
- Recurrent high-frequency conflict pairs were marked `domain_priority_pair` in `domain_priority` mode.
- This keeps architecture unchanged: same two-layer routing, only conflict resolution changes for selected pairs.

## Where domain priority was introduced
- Added curated pair-level priority for repeated conflicts such as:
  - `Упаковка -> Крепления и соединения`
  - `Материал -> Инструкция`
  - `Цена -> Оплата и расчёты`
  - `Доставка -> Запись на услугу`
  - `Расположение -> Номер`
  - `Загруженность -> Ожидание`
  - `Общее впечатление -> Поедаемость`

## Why
- Top overlap was dominated by a small set of repeated general-domain collisions.
- Narrow synonym cleanup alone gave only a small overlap drop.
- The main gain came from letting the more specific domain anchor win on those repeated collisions.
