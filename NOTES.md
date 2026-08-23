# World, Action & Reward Models — доучить VLA малым бюджетом демо

## Задание

**100 баллов.** Доучить VLA до новых задач малым бюджетом демонстраций.

### Цель

Современные VLA-модели (vision-language-action) обучаются на больших робо-датасетах
и обещают «обучить один раз, использовать в самых разных задачах». На практике узкое
место — цена адаптации: каждая новая задача требует новых телеоп-демонстраций, а
телеоп дорог и не скейлится. Цель — сдвигать эту кривую: тот же success за меньше
разметки, в пределе из одного лишь видео. Вторая цель — **оценка без реварда**: у
большинства реальных задач нет reward-функции, и мы учим модели измерять прогресс и
качество поведения прямо по видео.

С этого стоит начать: модель, дообученная на десятках задач одного робота в похожих
сценах, на новой задаче будет то справляться с полуслова, то оставаться беспомощной
даже после 25 демонстраций. Куда девается генерализация и что возвращает её дешевле
всего?

Главная метрика задания — **cost curve**: success на новой задаче в зависимости от
числа демонстраций $n \in \{0, 5, 10, 25\}$ (ноль значит zero-shot, по одной языковой
инструкции). Метод адаптации хорош ровно настолько, насколько он сдвигает эту кривую
влево. Правильного решения у задания нет: ваши выборы и их обоснования интересуют
не меньше, чем итоговые числа.

### Контекст

- **LIBERO** — стандартный бенчмарк адаптации VLA к новым задачам; используем через
  официальную интеграцию в LeRobot, там же описана установка; датасет —
  `HuggingFaceVLA/libero`.
- **SmolVLA** ($\approx 450$M) — модель, с которой вы работаете; базовый чекпойнт —
  `lerobot/smolvla_base`, дообучение на LIBERO — ваша Задача 1.
- **Сплит фиксированный**, чтобы решения были сравнимы: seen-часть — `libero_90`
  (можно разумное подмножество, если компьюта мало, но скажите об этом явно),
  целевые held-out задачи — первые три задачи `libero_goal` (`task_ids` 0–2).
- Полезные статьи: **OpenVLA** — адаптация VLA на большем масштабе; **LAPO** —
  латентные действия из видео без разметки (к бонусу A); **TimeRewarder** — сигнал
  прогресса из пассивного видео (к бонусу B); **Robometer** с открытым чекпойнтом
  `Robometer-4B-LIBERO` — готовая reward-модель (к бонусу B).

Пользуйтесь любыми библиотеками, фреймворками и AI-агентами: исходим из того, что
кодинг-агентами вы пользоваться будете, и финальный созвон устроен с учётом этого.

### Задача 1. Seen-претрен, zero-shot и предсказания (20 баллов)

Дообучите `smolvla_base` на seen-части. Есть несколько разумных путей: полный
`libero_90`, его подмножество под ваш компьют, готовый открытый LIBERO-файнтюн, если
найдёте такой. Выберите один и объясните почему.

Измерьте **точку 0**: success вашего seen-чекпойнта на целевых задачах по одной лишь
инструкции. Добавьте **контроль на язык**: прогоните те же эпизоды с инструкцией от
другой задачи. Если success не изменился — модель не читает инструкцию; указать на
это и разобраться ценнее, чем замолчать.

Зафиксируйте **бейзлайн**: наивный файнтюн вашего seen-чекпойнта на 5/10/25 демо
целевой задачи (берите первые по порядку в датасете, не выбирайте удачные), без
ухищрений. Это точка отсчёта для Задачи 2, менять её задним числом нельзя.

До любых дальнейших запусков запишите в отчёт **предсказания**: какой формы будет
ваша итоговая кривая и почему. Несбывшиеся предсказания не штрафуются: их разбор в
Задаче 5 обычно интереснее сбывшихся.

### Задача 2. Ваш метод адаптации (25 баллов)

Побейте свой бейзлайн на каждом из бюджетов 5/10/25. Как — на ваше усмотрение: LoRA
или полный файнтюн, своя голова действий, токенизация и чанкинг действий,
аугментации, претрен-микс из демо seen-задач. Каждый подход — это гипотеза: перед
запуском предскажите, поможет он сильнее при 5 демо или при 25, и проверьте.

**Жёсткие ограничения:** только данные LIBERO (среда нужна для оценки, не для сбора
нового); бюджет демо на целевую задачу строгий (5/10/25, первые по порядку в
датасете); демо seen-задач использовать можно, демо других целевых задач — нельзя.

**Итог задачи** — cost curve $0/5/10/25$, ваш метод против вашего бейзлайна: среднее
по трём целевым задачам, не меньше 20 эпизодов оценки на задачу и точку, минимум два
сида обучения.

### Задача 3. Разбор фейлов (15 баллов)

Поднимите реальные роллауты: три характерных провала с кадрами или видео. Для
каждого — ваша гипотеза, что именно ломается, и дизайн эксперимента, который эту
гипотезу отделил бы от альтернативной. Проводить эти эксперименты не обязательно,
оценивается дизайн.

### Задача 4. Бонус на выбор (20 баллов, выберите один)

- **A. Action-free данные.** Соберите пул сами: возьмите демо seen-задач и не
  используйте их действия, только видео. Можно ли извлечь из такого пула пользу для
  адаптации: латентные действия, inverse dynamics, представления? Это предельная
  форма всей постановки задания — учиться действовать из видео без разметки.
- **B. Оценка без реварда.** Выучите по видео собственный сигнал прогресса и
  отранжируйте им свои чекпойнты; сравните с готовой `Robometer-4B-LIBERO`: кто
  ранжирует ближе к настоящему success — ваш маленький критик или 4B-фаундейшен? И
  отдельно: если оптимизировать политику прямо на выученный сигнал, она станет лучше
  или научится его обманывать?
- **C. Длинный горизонт.** LIBERO-Long: многостадийные задачи. Помогает ли
  декомпозиция на сабтаски и планирование поверх них там, где плоский файнтюн
  ломается?

### Задача 5. Summary и обсуждение (20 баллов)

Три пункта, в любом порядке:

- **Предсказания против результатов:** где вы ошиблись и что это говорит о ваших
  исходных представлениях?
- **Кладбище идей:** два-три подхода, которые вы рассмотрели и отвергли, и почему.
- **Самое неожиданное наблюдение** за время работы и ваше объяснение.

### Что нужно сдать

1. Запускаемый репозиторий с кодом и README: setup, точные команды воспроизведения
   каждой задачи. Чистота и воспроизводимость кода — часть оценки.
2. Отчёт, **максимум 4 страницы**: предсказания (записанные до запусков), cost curve,
   разбор фейлов, результаты бонуса, ответы Задачи 5. Лимит жёсткий: читать будут
   ровно четыре страницы, остальное складывайте в приложение к репозиторию.

---

## Сетап

| поле | значение |
|---|---|
| модель | `lerobot/smolvla_base` |
| seen | официальный авторский `libero_90`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` |
| формат seen | LeRobot v3, 20 FPS, 256x256, две камеры, все 4500 эпизодов |
| target | официальный авторский `libero_goal`, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, первые три задачи |
| формат target | наша LeRobot v3-конверсия, 20 FPS, 256x256, две камеры, все 500 эпизодов |
| бюджеты | 0, 5, 10, 25 первых демонстраций каждой target-задачи |
| финальная оценка | 3 задачи, 20 или больше эпизодов, 2 или больше training seeds |
| локальная проверка | короткий запуск на MPS, только проверка кода и диагностики |
| полный запуск | A100 через ClearML |

Конверсия сохраняет 20 FPS из авторского HDF5. Изображения поворачиваются на 180 градусов, как в `LiberoProcessorStep`, затем увеличиваются с 128x128 до 256x256 через bicubic resize без crop. State равен `ee_states + gripper_states`, action переносится без изменения смысла. Числовые массивы записываются в `float32`, как требует обучение SmolVLA.

## Эксперименты

### Что логируем

- Один раз на запуск: dataset repo и revision, episode IDs, device, dtype, batch, workers, seed, optimizer, scheduler и число обучаемых параметров.
- На каждом шаге: loss и его части, gradient norm, LR выполненного шага, LR следующего шага, время и samples/s.
- Для обычной VLA: target и predicted action chunks на фиксированных кадрах, ошибка по 50 шагам, rollout success и видео.
- Для latent-метода: transition loss, cosine, нормы настоящего и предсказанного (z), actions из настоящего, предсказанного, нулевого и переставленного (z).

Гиперпараметры сохраняются для воспроизводимости, но не считаются диагностикой поведения.

### Предварительные прогоны

Один MPS seed, одна target-задача, 5 демо, 5 eval-эпизодов. Обычные методы учатся 100 шагов, контроль длительности 200, latent-фазы 50; warmup равен 10 шагам. Эти числа нужны для сравнения динамики за короткий локальный прогон.

- Наивный fine-tune обучает штатный action expert: 99.9M параметров, vision encoder заморожен.
- Удвоенная длительность использует ту же модель и те же 99.9M параметров, но 200 шагов вместо 100.
- Полное размораживание отключает expert-only и обучает 392.9M параметров, включая vision encoder; число шагов остаётся 100.
- LoRA замораживает базовые веса и обучает 1.49M adapter-параметров за 100 шагов.

| эксперимент | основание | код | статус | запуск | результат | диагностика |
|---|---|---|---|---|---|---|
| 1. Конверсия `libero_90` и `libero_goal` | данные из условия | [конвертер](tmp/convert_libero.py) | завершена | [seen](logs/libero_90_conversion.json), [target](logs/libero_goal_conversion.json) | 4500 seen и 500 target эпизодов; три target-задачи прочитаны | [кадры и pixel diff](runs/diagnostics/conversion_test.png) |
| 2. Явный цикл SmolVLA | проверка инфраструктуры | [цикл](vla/training.py), [test](tmp/smoke_training_boundary.py) | один шаг прошёл; checkpoint повторно загружен | [метрики](runs/boundary_explicit_loop/metrics.jsonl), [run](runs/boundary_explicit_loop/run.json) | loss 3.682; LR шага 1e-4 | loss, gradient norm, LR, action-loss components |
| 3. Seen-претрен | Задача 1 | [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_seen_pretrain/metrics.jsonl), [run](runs/preliminary_seen_pretrain/run.json) | loss 3.831 → 1.237 | [action chunks](runs/diagnostics/preliminary_seen_pretrain_actions.json) |
| 4. Zero-shot | Задача 1 | [очередь](run_preliminary.py) | завершён | [eval](eval_logs/preliminary_zero_shot_t0/eval_info.json) | success 0/5 | [action chunks](runs/diagnostics/preliminary_zero_shot_t0_actions.json), видео в eval |
| 5. Контроль с чужой инструкцией | Задача 1 | [очередь](run_preliminary.py) | завершён | [eval](eval_logs/preliminary_wrong_instruction_t0/eval_info.json) | success 0/5 | [action chunks](runs/diagnostics/preliminary_wrong_instruction_t0_actions.json), видео в eval |
| 6. Наивный fine-tune | baseline | [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_naive_finetune_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_naive_finetune_t0_n5_s0/eval_info.json) | loss 1.707 → 0.685; success 0/5 | [action chunks](runs/diagnostics/preliminary_naive_finetune_t0_n5_s0_actions.json) |
| 7. Удвоенная длительность | контроль для mix | [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_longer_finetune_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_longer_finetune_t0_n5_s0/eval_info.json) | loss 1.707 → 0.404; success 0/5 | [action chunks](runs/diagnostics/preliminary_longer_finetune_t0_n5_s0_actions.json) |
| 8. Полное размораживание | Задача 2 | [метод](vla/methods.py), [test](tmp/full_finetune_boundary.py) | завершён | [метрики](runs/preliminary_full_finetune_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_full_finetune_t0_n5_s0/eval_info.json) | loss 1.707 → 0.749; success 0/5 | [action chunks](runs/diagnostics/preliminary_full_finetune_t0_n5_s0_actions.json) |
| 9. Подмешивание seen | Задача 2, `OWNER_NOTES.md` | [data](vla/data.py), [очередь](run_preliminary.py) | завершён, target не прореживался | [метрики](runs/preliminary_mix_seen_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_mix_seen_t0_n5_s0/eval_info.json) | loss 1.352 → 0.806; success 0/5 | [action chunks](runs/diagnostics/preliminary_mix_seen_t0_n5_s0_actions.json) |
| 10. LoRA r=32, LR 1e-3 | Задача 2 | [метод](vla/methods.py), [test](tmp/lora_training_boundary.py) | завершён | [метрики](runs/preliminary_lora_r32_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_lora_r32_t0_n5_s0/eval_info.json) | loss 1.707 → 0.711; success 0/5 | [action chunks](runs/diagnostics/preliminary_lora_r32_t0_n5_s0_actions.json) |
| 11. Action chunk 10 | Задача 2 | [метод](vla/methods.py), [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_chunk_10_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_chunk_10_t0_n5_s0/eval_info.json) | loss 2.009 → 1.075; success 0/5 | [action chunks](runs/diagnostics/preliminary_chunk_10_t0_n5_s0_actions.json) |
| 12. Аугментации изображений | Задача 2 | [метод](vla/methods.py), [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_image_augmentations_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_image_augmentations_t0_n5_s0/eval_info.json) | loss 1.713 → 0.693; success 0/5 | [action chunks](runs/diagnostics/preliminary_image_augmentations_t0_n5_s0_actions.json), [кадры](runs/diagnostics/augmentations/metadata.json) |
| 13. Latent transition и decoder с seen actions | `OWNER_NOTES.md` | [policy](vla/modeling_latent_smolvla.py), [очередь](run_preliminary.py) | завершён | [transition](runs/preliminary_latent_transition/metrics.jsonl), [decoder](runs/preliminary_latent_seen_decoder/metrics.jsonl), [eval](eval_logs/preliminary_latent_seen_actions_t0_n5_s0/eval_info.json) | transition loss 1.989 → 1.167; success 0/5 | [latent](runs/diagnostics/preliminary_latent_transition_transitions.json), [actions](runs/diagnostics/preliminary_latent_seen_actions_t0_n5_s0_actions.json) |
| 14. Latent transition без seen actions | Bonus A | [policy](vla/modeling_latent_smolvla.py), [очередь](run_preliminary.py) | завершён | [метрики](runs/preliminary_latent_video_only_t0_n5_s0/metrics.jsonl), [eval](eval_logs/preliminary_latent_video_only_t0_n5_s0/eval_info.json) | loss 0.289 → 0.273; success 0/5 | [action controls](runs/diagnostics/preliminary_latent_video_only_t0_n5_s0_actions.json) |
| 15. Tiny-set overfit latent | capacity и wiring gate перед A100 | [диагностика](tmp/latent_tiny_overfit.py) | заблокирован на MPS | метрик нет | Metal завершает процесс `SIGABRT` внутри `Linear`; метод не оценён | transition cosine, action MAE для true, predicted, zero и reversed latent |

У всех вариантов training loss снизился, но каждый получил success 0/5. Всего получено 0 успехов в 55 предварительных rollout-эпизодах. Seen-претрен здесь длился только 100 шагов, поэтому этот отсев проверяет код и короткую динамику, но не оценивает полноценную адаптацию после обучения на `libero_90`.

- На трёх фиксированных кадрах из target-демонстраций action MAE равен 0.136 у fine-tune на 200 шагов, 0.179 у полного размораживания и 0.196 у наивного fine-tune на 100 шагов. Это измеряет запоминание обучающих демонстраций, а не rollout-обобщение.
- LoRA, chunk 10, аугментации и подмешивание seen не улучшили action MAE относительно наивного fine-tune в этом коротком прогоне: 0.208, 0.187, 0.199 и 0.210 соответственно.
- У latent transition средний cosine с настоящим visual-token transition равен 0.0003. Перестановка 50 предсказанных transition-шагов меняет decoded actions в среднем на 0.000003. Текущая latent-ветка не показывает, что выучила направление перехода или порядок будущих шагов; увеличивать её бюджет без отдельного разбора нельзя.

Перед A100 запускаем tiny-set overfit latent-ветки. Если predictor не запоминает три фиксированных перехода и decoder не начинает зависеть от latent, большой latent-прогон не запускается.

На A100 остаются обязательный наивный baseline, подмешивание seen и LoRA. Chunk 10, полное размораживание, удвоенная длительность и аугментации исключены владельцем. Наивный fine-tune не является кандидатом, но остаётся обязательной точкой отсчёта из Задачи 1. Latent Bonus A добавляется только после успешного tiny-set overfit.

### Финальные прогоны

Финальная матрица использует три target-задачи, бюджеты 5/10/25, training seeds 0 и 1 и 20 eval-эпизодов на каждую точку. Все методы стартуют из одного полного seen-checkpoint. Target-демо всегда берутся первыми по порядку.

#### Фиксированные настройки A100

| поле | значение |
|---|---|
| устройство | CUDA, BF16 |
| batch и workers | 32, 8 |
| action chunk | 50 предсказанных и 50 исполняемых действий |
| flow matching | 10 шагов интеграции на каждом новом chunk |
| seen-претрен | все 4500 эпизодов `libero_90`, 30 000 optimizer steps, seed 0 |
| target fine-tune | 300 optimizer steps на демонстрацию: 1500, 3000 и 7500 шагов |
| baseline и mix optimizer | AdamW, LR 1e-4, betas 0.9/0.95, eps 1e-8, weight decay 1e-10, gradient clip 10 |
| baseline и mix scheduler | 100 warmup steps, cosine decay до 2.5e-6 |
| LoRA | r=32, alpha=32, LR 1e-3, 100 warmup steps, cosine decay до 1e-5 |
| LoRA targets | q/v projections action expert, state projection, action input/output projections, две time projections |
| mix | столько дополнительных optimizer steps, чтобы число target-примеров совпало с baseline |
| оценка | `libero_goal` task IDs 0, 1, 2; init states из LIBERO; eval seed 1000; 20 эпизодов; максимум 300 env steps |

Seen-претрен использует тот же AdamW с LR 1e-4, 1000 warmup steps и cosine decay до 2.5e-6. Zero-shot и wrong instruction ничего не обучают. Оба оценивают один seen-checkpoint. Wrong instruction меняет только текст задачи и проверяет, влияет ли язык на поведение.

| эксперимент | основание | код | статус | запуск | результат | диагностика |
|---|---|---|---|---|---|---|
| 1. Seen-претрен | Задача 1 | `run_final.py` | код не написан |  |  | loss и фиксированные action chunks |
| 2. Zero-shot | точка 0 | `run_final.py` | код не написан |  |  | 3 задачи, rollout success и видео |
| 3. Wrong instruction | контроль языка | `run_final.py` | код не написан |  |  | тот же checkpoint, init states и eval seeds |
| 4. Наивный fine-tune cost curve | обязательный baseline | `run_final.py` | код не написан |  |  | 18 training cells, action chunks и видео |
| 5. Подмешивание seen | кандидат Задачи 2 | `run_final.py` | код не написан |  |  | 18 cells и одинаковая target-экспозиция |
| 6. LoRA r=32 | кандидат Задачи 2 | `run_final.py` | код не написан |  |  | 18 cells и фактические adapter targets |
| 7. Latent Bonus A | Задача 4 | `run_final.py` | ждёт tiny-set overfit |  |  | cost curve, cosine, zero и reversed latent controls |
| 8. Три характерных провала | Задача 3 | [rollouts](rollouts.py), [HTML](viz.py) | ждёт финальные rollout-ы |  |  | видео и различающий эксперимент для каждого фейла |

### Что показывают старые proxy-прогоны

Все семь прогонов использовали `lerobot/libero`, `smolvla_base` без seen-претрена, пять первых демонстраций одной задачи `libero_goal`, один training seed и 20 eval-эпизодов. Dataset revision не был закреплён в конфиге. Это отсев реализаций, а не результаты задания.

`libero_object` совместим по роботу и тензорам: две камеры 256x256, state размерности 8 и action размерности 7. Но пять подмешанных эпизодов относятся к узким задачам `pick object → basket`, тогда как `libero_90` содержит 90 более разнообразных seen-задач. Старый набор также имеет 10 FPS, а согласованный `libero_90` конвертируется в 20 FPS. Поэтому абсолютные success и результат чанкинга не переносятся.

Подмешивание было честным по target-экспозиции. В пяти target-эпизодах 682 кадра, в пяти `libero_object` эпизодах 708 кадров; 3000 шагов смеси дают 98.1% числа target-примеров относительно 1500 шагов без смеси. Контроль без смеси также обучался 3000 шагов. Результат 0.80 против 0.50 делает подмешивание первым кандидатом на повтор после seen-претрена, но одного seed и одной задачи недостаточно для выбора итогового метода.

LoRA действительно обучалась с эффективным LR 1e-3 и 1.49M обучаемых параметров. Результат 0.30 взят из `tmp/trick_lora_lr_eval2.log`, записанного после финального checkpoint; более ранний eval на незавершённом checkpoint недействителен. Полное размораживание обучало 393M параметров, дефолтный fine-tune 100M. Старый текст сохранён в `tmp/legacy/NOTES_before_rewrite.md`.

## Проверка данных и запуска

- Конвертер останавливается при несовпадении размеров массивов, пропавших файлах, NaN и неверном числе эпизодов. Это проверка корректности, а не диагностика эксперимента.
- Каждый запуск сохраняет применённые dataset revision, task и episode IDs, device, dtype, batch size, workers, seed, optimizer, LR, scheduler, обучаемые параметры и новый checkpoint. Это проверяет, что LeRobot не перезаписал настройки своим preset.
- Методы сравниваются на одинаковых task IDs, init states, eval seeds и числе эпизодов. Короткий отсев помечается отдельно от финального результата.

## Диагностика поведения

- Loss показывается вместе с predicted и target action chunks на фиксированных кадрах. Один loss не показывает, какое действие модель научилась делать.
- Контроль языка сравнивает action chunks при одной observation и двух инструкциях. Rollout success проверяет, влияет ли разница на поведение в среде.
- Гипотеза забывания проверяется на одних и тех же seen и target кадрах до и после target fine-tune. Сравниваются action chunks и роллауты.
- Для метода представлений на одних фиксированных парах показываются настоящий \(z_t\), предсказанный \(\hat z_t\), action из настоящего \(z_t\) и action из \(\hat z_t\). Обнуление и перестановка \(z\) проверяют, использует ли decoder этот код. Проекции добавляются только после просмотра самих тензоров.
- Перестановка 50 latent-шагов не требовалась заданием и не заменяет rollout success. Это наш causal control для выбранной архитектуры: если перестановка будущих переходов не меняет соответствующие actions, весь latent-путь не использует временной порядок chunk.
- HTML строится только из JSON реальных запусков. Отсутствующие значения показываются как `not recorded`.

## Согласованный метод латентного изменения кадров

Замороженный visual encoder SmolVLA кодирует два соседних наблюдения:

$$
h_t=\phi(o_t),\qquad h_{t+1}=\phi(o_{t+1}),\qquad z_t=h_{t+1}-h_t.
$$

Здесь \(o_t\) обозначает кадры камер, \(h_t\) их visual tokens, а \(z_t\) изменение этих tokens между соседними моментами. Пиксельную разницу не используем. Берём только пару \(t,t+1\), без дальнего \(t+k\) и без VQ-дискретизации.

По seen-видео predictor учится восстанавливать \(z_t\) из текущего наблюдения и инструкции:

$$
P(h_t,\text{instruction})\rightarrow \hat z_t.
$$

Прямой decoder связывает латентное изменение с управлением:

$$
D(z_t)\rightarrow a_t.
$$

В основном методе decoder сначала учится на actions из seen-демонстраций, затем адаптируется на разрешённых 5/10/25 target-демонстрациях. В бонусе A predictor получает те же seen-видео, но actions seen-задач полностью скрыты; decoder впервые видит actions только в разрешённых target-демонстрациях. На инференсе работают \(P(h_t,\text{instruction})\) и \(D(\hat z_t)\), следующий кадр недоступен.

Берём последние замороженные visual tokens, которые vision encoder передаёт в VLM. Predictor получает token текущего кадра, средний language embedding, номер будущего шага и номер камеры; общий двухслойный MLP предсказывает 50 последовательных token-deltas. Decoder учится на настоящих разностях соседних кадров, усредняет tokens внутри каждой камеры, конкатенирует две камеры и линейно выдаёт штатный chunk из 50 actions. На rollout будущих кадров нет, поэтому decoder получает предсказанные разности; все 50 actions исполняются до следующего планирования, как в baseline SmolVLA.

### Отфильтрованные варианты

- CLAP-подобный encoder \(a\to z\) не делаем. И он, и прямой decoder \(z\to a\) используют action labels для одной связи; contrastive направление добавляет модель и loss, но не проверяет другой вопрос.
- Action bottleneck \((o,\text{text})\to z\to a\) является общей схемой выбранного метода, а не отдельным экспериментом.
- Dynamics-aware representation является целью того же \(z_t=h_{t+1}-h_t\), а не вторым методом. Отдельный world-model loss пока не добавляем.
- VQ и дальний переход \(t+k\) не используем по решению владельца. Работаем с непрерывной разницей соседних visual tokens.
- Разность robot states может сильнее коррелировать с action, но для Bonus A она использует proprioception вместо заявленного видео и почти восстанавливает управляющий сигнал. Её можно оставить только как контроль верхней границы.
- Progress representation и TimeRewarder относятся к Bonus B. Они измеряют выполнение задачи, а не удешевляют привязку движения к action.
- Общий loss из action, temporal, transition и retain без отдельной проверки каждого слагаемого не используем. Он не позволит понять, что дало эффект.

Предсказание владельца: выигрыш video representation должен быть максимален на 5 демо, а к 25 демо наивный fine-tune может частично догнать. Старый proxy ставит подмешивание первым кандидатом на повтор; он не позволяет заранее объявить его победителем на правильном split.

## Открытые решения

1. Для подмешивания сохраняется правило: число показов target-примеров совпадает с наивным fine-tune, поэтому число optimizer steps увеличивается пропорционально размеру смеси.

## Предсказания

Заполнить до соответствующих запусков. Старые предсказания относились к неверному seen-набору и автоматически не переносятся.
