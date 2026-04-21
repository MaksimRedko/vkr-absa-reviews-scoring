# Vocabulary Reproducibility Log

## Experiment
- Name: `phase1_step1_physical_goods_draft`
- Goal: собрать первый broad domain sub-vocabulary для `physical_goods` без использования gold labels.
- Baseline: `src/vocabulary/universal_aspects_v1.yaml`
- Variable changed: добавлен только `src/vocabulary/domain/physical_goods.yaml`
- Metric to improve: `coverage(core)` vs `coverage(core + physical_goods)` для категории `physical_goods`

## Design Constraints For This Revision
- Не использовать маркетплейсы и live browsing по карточкам товаров.
- Не смешивать узкие микродомены внутри одной category.
- Собрать только broad-aspects слой для материальных товаров.
- Оптимизировать coverage, а не полноту описания мира.
- Оставить только аспекты, которые повторяются на нескольких типах physical goods.

## Primary Sources
1. `https://support.google.com/merchants/answer/6324436?hl=en`
   Официальная Google product taxonomy reference. Использована как стабильный category/taxonomy source, подтверждающий, что physical goods охватывает широкий набор товарных веток, а не одну нишу.

2. `https://aclanthology.org/S14-2004/`
   SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Использовано как академический reference point для принципа aspect-schema design: аспекты должны быть согласованы на уровне домена и не превращаться в смесь несопоставимых нишевых наборов.

## Auxiliary Sources
1. `https://ru.wikipedia.org/wiki/Одежда`
   Использовано только как вспомогательный источник лексики для общих physical-goods аспектов, связанных с эксплуатацией изделия.

2. `https://ru.wikipedia.org/wiki/Шитьё`
   Использовано только как вспомогательный источник терминов для сборки, инструкции, креплений и обслуживания изделий.

3. `https://simple.wikipedia.org/wiki/Consumer_goods`
   Использовано только как вспомогательное описание класса consumer goods / final goods.

## Included Aspects For `physical_goods`

### `assembly_installation`
- Canonical: `Сборка и установка`
- Why not core: это не универсальная характеристика всех доменов, а стадия ввода physical good в использование.
- Why broad for physical_goods: аспект переносится на мебель, аксессуары, наборы, электронику, спортивные и домашние товары, которые нужно собрать или установить.

### `instructions_manual`
- Canonical: `Инструкция`
- Why not core: инструкция релевантна не всем отзывным доменам и относится к информационному сопровождению физического изделия.
- Why broad for physical_goods: руководство по использованию важно для бытовых товаров, наборов, техники, игрушек, хобби-товаров и инструментов.

### `fit_applicability`
- Canonical: `Подходит / не подходит`
- Why not core: это не то же самое, что общее `Соответствие`; здесь сигнал про практическую применимость, посадку или совместимость в использовании.
- Why broad for physical_goods: аспект переносится на одежду, аксессуары, комплектующие, чехлы, расходники и другие товары, где пользователь оценивает, подошёл ли товар по задаче, форме или размерности.

### `ease_of_use`
- Canonical: `Удобство использования`
- Why not core: в этой ревизии аспект понимается уже, чем общий `Удобство` из core: не общая приятность, а именно эксплуатационное удобство physical good.
- Why broad for physical_goods: аспект переносится на инструменты, аксессуары, одежду, домашние товары и устройства, где важна usability в реальном использовании.

### `care_maintenance`
- Canonical: `Уход и обслуживание`
- Why not core: это не общая удобность и не обслуживание как сервис, а требования к поддержанию физического изделия в рабочем состоянии.
- Why broad for physical_goods: уход релевантен одежде, обуви, домашним товарам, технике, посуде и другим материальным товарам длительного использования.

### `connections_fasteners`
- Canonical: `Крепления и соединения`
- Why not core: это более предметный аспект, чем общее качество, и относится именно к устройству physical object.
- Why broad for physical_goods: крепления и соединения встречаются у одежды, сумок, мебели, аксессуаров, контейнеров, домашних и спортивных товаров.

### `repairability`
- Canonical: `Ремонтопригодность`
- Why not core: это отдельная характеристика жизненного цикла physical good, а не общий sentiment dimension для всех доменов.
- Why broad for physical_goods: аспект переносится на durable goods, технику, мебель, инструменты и изделия с заменяемыми деталями.

## Candidate Aspects Rejected As Core Duplicates
- `Качество`
- `Цена`
- `Доставка` / `Логистика`
- `Упаковка`
- `Размер`
- `Материал`
- `Цвет`
- `Запах`
- `Функциональность`
- `Ассортимент` / `Комплектация`
- `Соответствие`
- `Удобство`
- `Внешний вид`
- `Надёжность`

## Candidate Aspects Rejected As Too Narrow / Niche
- `Текст и печать` -> слишком завязано на книги и печатную продукцию
- `Содержание` -> слишком завязано на книги, наборы и content-bearing goods
- `Питание` -> слишком завязано на электронику и battery-powered devices
- `Зарядка` -> слишком завязано на rechargeable electronics
- `Удилище` -> рыболовный микродомен
- `Катушка` -> рыболовный микродомен
- `Леска` -> рыболовный микродомен
- `Приманка` -> рыболовный микродомен
- `Крючок и оснастка` -> рыболовный микродомен
- `Швы и строчки` -> слишком тяготеет к одежде и текстилю
- `Крой и выкройка` -> слишком тяготеет к одежде и рукоделию

## Candidate Aspects Deferred As Lower-Priority For Coverage
- `Состояние` -> полезный broad aspect, но на этом шаге отложен ради более частотных сигналов про подходящесть и использование
- `Безопасность использования` -> broad aspect, но ожидаемо реже встречается как массовый signal в текущем physical_goods слое

## Notes
- В этой ревизии Wikipedia используется только как auxiliary source для лексической стабилизации формулировок, а не как основной источник архитектуры словаря.
- Основной методологический принцип этой ревизии: broad physical-goods vocabulary должен покрывать повторяющиеся признаки материальных товаров, а не частные признаки узких ниш.

---

## Experiment
- Name: `phase1_step2_hospitality_draft`
- Goal: собрать broad domain sub-vocabulary для `hospitality` без использования gold labels.
- Baseline: `src/vocabulary/universal_aspects_v1.yaml`
- Variable changed: добавлен только `src/vocabulary/domain/hospitality.yaml`
- Metric to improve: `coverage(core)` vs `coverage(core + hospitality)` для категории `hospitality`

## Design Constraints For This Revision
- Не использовать маркетплейсы и live browsing по карточкам товаров.
- Использовать только стабильные и воспроизводимые hotel/lodging sources.
- Собрать только broad hospitality aspects.
- Не дублировать core vocabulary.
- Оптимизировать coverage, а не полноту описания мира.

## Primary Sources
1. `https://www.hotelstars.eu/for-guests/criteria-at-a-glance/`
   Официальные критерии Hotelstars Union для классификации отелей. Использованы как основной источник broad hotel dimensions: room, bed, bathroom, breakfast, reception/check-in, internet, parking.

2. `https://www.hotelstars.eu/criteria/`
   Расширенная версия критериев Hotelstars Union. Использована для проверки, что выбранные аспекты устойчивы и повторяются на уровне hotel category system, а не одной площадки.

3. `https://developers.google.com/hotels/hotel-content/proto-reference/lodging-proto`
   Официальная Google lodging schema. Использована как стабильный category description source для hotel-level and room-level attributes, включая internet, parking, breakfast, check-in/check-out и room-related fields.

4. `https://schema.org/Hotel`
   Канонический открытый словарь для hotel/lodging entities. Использован для подтверждения broad hotel business-level aspects и stay-related metadata.

5. `https://schema.org/LodgingReservation`
   Канонический открытый словарь для reservation layer. Использован для обоснования отдельного аспекта `Бронирование`.

## Included Aspects For `hospitality`

### `room`
- Canonical: `Номер`
- Why not core: это не общая инфраструктура и не общее качество, а центральный объект гостиничного опыта.
- Why broad hospitality: номер обсуждается практически во всех hotel/lodging reviews, независимо от подкласса отеля.

### `bed_sleep`
- Canonical: `Кровать и сон`
- Why not core: это уже, чем общее удобство; это специфический lodging signal про кровать, матрас, подушку и качество сна.
- Why broad hospitality: качество сна и кровати повторяется в hotel review templates и относится к базовому гостиничному опыту.

### `bathroom`
- Canonical: `Ванная комната`
- Why not core: это не то же самое, что общая чистота; здесь отдельный physical space внутри hotel stay.
- Why broad hospitality: ванная/душ/туалет - стандартная повторяющаяся часть проживания в отеле.

### `checkin_checkout`
- Canonical: `Заселение и выезд`
- Why not core: это не просто скорость и не просто сервис; это отдельный этап гостиничного journey.
- Why broad hospitality: check-in/check-out присутствует практически во всех lodging scenarios.

### `breakfast_food`
- Canonical: `Завтрак и питание`
- Why not core: это не общий service signal, а отдельный food-related слой гостиничного продукта.
- Why broad hospitality: завтрак и базовое питание - один из самых типовых hotel review dimensions.

### `wifi_internet`
- Canonical: `Wi-Fi и интернет`
- Why not core: это узкий operational amenity, который не нужен как общий аспект для всех доменов.
- Why broad hospitality: интернет стабильно фигурирует в hotel criteria и review templates как стандартная amenity-category.

### `parking`
- Canonical: `Парковка`
- Why not core: парковка слишком предметна для ядра и относится к lodging/travel context.
- Why broad hospitality: parking - типовой аспект городских и дорожных hotel stays.

### `reservation_booking`
- Canonical: `Бронирование`
- Why not core: бронирование относится к hotel reservation flow, а не к общему sentiment layer across domains.
- Why broad hospitality: booking/reservation - стандартный этап взаимодействия с гостиницей до фактического проживания.

### `noise_insulation`
- Canonical: `Шумоизоляция`
- Why not core: это не просто комфорт или атмосфера, а отдельный аспект acoustic privacy during stay.
- Why broad hospitality: шум и слышимость регулярно описывают качество проживания в отелях разных типов.

## Candidate Aspects Rejected As Core Duplicates
- `Обслуживание` / `персонал`
- `Чистота`
- `Расположение`
- `Цена`
- `Скорость`
- `Удобство` / `комфорт`
- `Атмосфера`
- `Безопасность`
- `Инфраструктура` / `оснащение`
- `Температура`

## Candidate Aspects Rejected As Too Narrow / Niche
- `Спа` -> слишком завязано на wellness подтип
- `Бассейн` -> не universal для broad hospitality
- `Трансфер` -> слишком частный service add-on
- `Мини-бар` -> узкий room-service micro-aspect
- `Пляж` -> релевантно только курортным объектам
- `Лыжное хранение` -> сезонный niche
- `Конференц-зал` -> business-hotel microdomain
- `Анимация` -> resort microdomain

## Candidate Aspects Rejected As Poorly Defensible
- `Роскошь` -> слишком расплывчатый и marketing-like label
- `Вау-эффект` -> неоперационализируемый аспект
- `Инстаграмность` -> незащитимый ad-hoc label
- `Статусность` -> неустойчивый и сильно интерпретируемый сигнал

---

## Experiment
- Name: `phase1_step3_services_draft`
- Goal: собрать broad domain sub-vocabulary для `services` без использования gold labels.
- Baseline: `src/vocabulary/universal_aspects_v1.yaml`
- Variable changed: добавлен только `src/vocabulary/domain/services.yaml`
- Metric to improve: `coverage(core)` vs `coverage(core + services)` для категории `services`

## Design Constraints For This Revision
- Не использовать маркетплейсы и live browsing по карточкам товаров.
- Собрать только broad aspects сервисного процесса и взаимодействия с клиентом.
- Не копировать `hospitality`-specific layer.
- Не дублировать core vocabulary.
- Оптимизировать coverage, а не полноту описания мира.

## Primary Sources
1. `https://developers.google.com/actions-center/verticals/reservations/e2e/reference/feeds/services-feed`
   Официальная service schema Google Actions Center. Использована как основной источник broad service-process concepts: service, scheduling rules, booking, update booking, cancellation, direct payment, support for waitlist and service attributes.

2. `https://doi.org/10.1177/002224298504900403`
   Parasuraman, Zeithaml, Berry (1985), conceptual model of service quality. Использован как академический источник для broad service-interaction dimensions: communication, competence, responsiveness, problem handling, customer-facing process.

3. `https://aclanthology.org/S16-1002/`
   SemEval-2016 Task 5: Aspect Based Sentiment Analysis. Использовано как академический reference point для aspect-schema design: аспектный слой должен быть доменно согласованным и воспроизводимым.

## Auxiliary Sources
1. `https://en.wikipedia.org/wiki/SERVQUAL`
   Использовано только как вспомогательное summary-описание service-quality dimensions и исторического перехода от competence / communication / responsiveness к operational service dimensions.

## Included Aspects For `services`

### `appointment_scheduling`
- Canonical: `Запись и запись на приём`
- Why not core: запись - это не общий service sentiment, а отдельный этап сервисного процесса.
- Why broad services: запись переносится между клиниками, салонами, сервисными центрами, консультациями и другими booking-based services.
- Why not hospitality overlap: в `hospitality` бронирование относится к lodging stay, а здесь речь о записи на услугу/слот выполнения.

### `waiting_process`
- Canonical: `Ожидание`
- Why not core: это уже, чем общая скорость; здесь конкретно waiting as part of service journey.
- Why broad services: ожидание встречается в очередях, обработке обращений, приёме, обслуживании и support flows.
- Why not hospitality overlap: в `hospitality` ожидание вторично к stay experience, а здесь это центральный operational service signal.

### `communication_updates`
- Canonical: `Информирование и коммуникация`
- Why not core: это не просто обслуживание, а отдельный канал передачи информации клиенту.
- Why broad services: информирование переносится между support, booking, консультациями, платформами и офлайн-сервисами.
- Why not hospitality overlap: это не hotel-stay amenity, а общая клиентская коммуникация в service process.

### `support_help`
- Canonical: `Поддержка`
- Why not core: поддержка - более узкий аспект, чем общее обслуживание.
- Why broad services: support channel повторяется во множестве service domains, особенно там, где есть обращения и сопровождение клиента.
- Why not hospitality overlap: это не guest-stay layer, а dedicated help/support interaction.

### `problem_resolution`
- Canonical: `Решение проблемы`
- Why not core: это не общее качество и не просто скорость; это outcome service recovery.
- Why broad services: аспект переносится между банкингом, телекомом, платформами, клиентским сервисом и очными услугами.
- Why not hospitality overlap: в `hospitality` фокус на проживании, а здесь на устранении клиентской проблемы как service outcome.

### `competence_professionalism`
- Canonical: `Компетентность`
- Why not core: это уже, чем общее обслуживание, и описывает знание и профессиональную состоятельность исполнителя.
- Why broad services: компетентность релевантна почти всем услугам, где клиент оценивает экспертность сотрудника.
- Why not hospitality overlap: это не room/stay layer и не hotel facility aspect, а общесервисная оценка исполнителя.

### `payment_billing`
- Canonical: `Оплата и расчёты`
- Why not core: это не просто цена, а процесс платежа, списаний, чеков и расчётов.
- Why broad services: оплата и расчёты повторяются в подписках, сервисных платформах, услугах по записи и клиентском обслуживании.
- Why not hospitality overlap: это не lodging-specific billing, а общий service transaction flow.

### `cancellation_refund`
- Canonical: `Отмена и возврат`
- Why not core: это отдельный service-policy/process layer, а не общая цена или удобство.
- Why broad services: отмена, перенос и возврат встречаются в большом числе service workflows.
- Why not hospitality overlap: здесь фокус на service cancellation/refund policy broadly, а не на hotel booking only.

## Candidate Aspects Rejected As Core Duplicates
- `Обслуживание` / `сервис`
- `Скорость`
- `Удобство`
- `Расположение`
- `Чистота`
- `Цена`
- `Надёжность`
- `Безопасность`
- `Атмосфера`
- `Загруженность`

## Candidate Aspects Rejected As Too Narrow / Niche
- `Модерация` -> platform-specific microdomain
- `Верификация` -> platform/fintech-specific microdomain
- `Подписка` -> subscription-service microdomain
- `Тарифы` -> too tied to telecom/financial/platform domains
- `Примерка` -> narrow commerce-service hybrid
- `Выдача багажа` -> transport-service microdomain
- `Таможня` -> airport-specific microdomain
- `Перрон` -> station-specific microdomain

## Candidate Aspects Rejected As Too Overlapping With Hospitality
- `Номер`
- `Кровать`
- `Ванная комната`
- `Заселение и выезд`
- `Завтрак и питание`
- `Wi-Fi и интернет`
- `Парковка`
- `Шумоизоляция`
- `Бронирование` -> в lodging sense оставлено только в hospitality

---

## Experiment
- Name: `phase1_step4_consumables_draft`
- Goal: собрать компактный broad domain sub-vocabulary для `consumables` без использования gold labels.
- Baseline: `src/vocabulary/universal_aspects_v1.yaml`
- Variable changed: добавлен только `src/vocabulary/domain/consumables.yaml`
- Metric to improve: `coverage(core)` vs `coverage(core + consumables)` для категории `consumables`

## Design Constraints For This Revision
- Не использовать маркетплейсы и live browsing по карточкам товаров.
- Не дублировать core vocabulary.
- Учитывать высокий baseline категории и добавлять только аспекты с потенциальным новым покрытием.
- Собрать только broad food/feed aspects, переносимые между разными consumable goods.

## Primary Sources
1. `https://www.aafco.org/consumers/understanding-pet-food/reading-labels/`
   Официальное описание pet food label semantics. Использовано как основной источник для broad consumable aspects: nutritional adequacy, feeding directions, calories.

2. `https://www.aafco.org/consumers/understanding-pet-food/`
   Официальное обзорное описание pet food regulation and consumer-facing dimensions. Использовано для стабилизации broad layer вокруг suitability, feeding, handling and nutritional interpretation.

3. `https://aclanthology.org/S14-2004/`
   SemEval-2014 Task 4: Aspect Based Sentiment Analysis. Использовано как академический reference point для aspect-schema design.

4. `https://aclanthology.org/S16-1002/`
   SemEval-2016 Task 5: Aspect Based Sentiment Analysis. Использовано как дополнительный академический reference point для воспроизводимого aspect design.

## Included Aspects For `consumables`

### `palatability_acceptance`
- Canonical: `Поедаемость`
- Why not core: это не то же самое, что `вкус`; аспект шире и описывает фактическое принятие consumable продукта при употреблении.
- Why broad for consumables: поедаемость релевантна и food, и pet food, и не привязана к одной нише.
- Why not niche: это один из самых частых user-facing signals для consumable products.

### `digestibility_tolerance`
- Canonical: `Усвояемость и переносимость`
- Why not core: это не `качество` и не `состав`; это post-consumption effect.
- Why broad for consumables: переносимость и усвоение применимы к еде, кормам и другим ingestible products.
- Why not niche: это типовой аспект для review о consumables, особенно при регулярном употреблении.

### `nutritional_adequacy`
- Canonical: `Питательная ценность`
- Why not core: это не просто состав ингредиентов, а функциональная nutritional adequacy продукта.
- Why broad for consumables: питательная ценность применима к еде, кормам, диетическим и everyday consumables.
- Why not niche: это стандартный consumer-facing аспект, закреплённый в label semantics и product choice.

### `caloric_satiety`
- Canonical: `Калорийность и сытость`
- Why not core: это уже, чем общее качество, и не сводится к вкусу или составу.
- Why broad for consumables: калорийность и насыщаемость релевантны многим food/feed products.
- Why not niche: аспект типичен для выбора продуктов питания и кормов, особенно в weight-control contexts.

### `feeding_portion`
- Canonical: `Норма кормления`
- Why not core: это не просто удобство и не состав, а guidance/usage layer consumable product.
- Why broad for consumables: нормы употребления и порционирование встречаются у еды, кормов, добавок и других consumables.
- Why not niche: это повторяющийся practical aspect, закреплённый в official labeling requirements.

## Candidate Aspects Rejected As Core Duplicates
- `Качество`
- `Цена`
- `Запах`
- `Состав`
- `Вкус`
- `Свежесть`
- `Срок годности`
- `Упаковка`
- `Логистика`
- `Ассортимент`

## Candidate Aspects Rejected As Too Narrow / Niche
- `Хруст гранул` -> слишком tied to dry pet food texture details
- `Влажность корма` -> слишком product-form specific
- `Пена напитка` -> напиточный microdomain
- `Острота` -> cuisine-specific narrow taste dimension
- `Порода животного` -> слишком narrow targeting axis, не aspect
- `Лечебный эффект` -> слишком близко к claim/medical niche

## Candidate Aspects Rejected As Poorly Defensible
- `Любовь питомца` -> слишком антропоморфный label
- `Премиальность` -> marketing-like, а не operational aspect
- `Натуральность` -> слишком расплывчато без привязки к формальной label semantics
