# Адаптация SmolVLA к новым задачам LIBERO

## 1. Постановка, данные и предсказания

Мы проверяем, сколько target-демонстраций нужно SmolVLA после претренинга на `libero_90`.

Seen-претрен использует `[FINAL SEEN STEPS]` шагов на всех 4500 эпизодах. Target содержит первые 5, 10 или 25 демонстраций каждой из первых трёх задач `libero_goal`.

Каждая финальная точка содержит `[FINAL N]` роллаутов. Мы используем `[FINAL SEEDS]` training seeds и одинаковые eval seeds.

До запусков мы ожидали `[PREDICTION ABOUT ZERO-SHOT]`.

Для наивного fine-tune мы ожидали `[PREDICTION ABOUT BASELINE CURVE]`.

Мы ожидали, что video representation сильнее всего поможет при 5 демо. К 25 демо наивный fine-tune должен частично сократить разрыв.

`[FIGURE: split с suite, task IDs, episode budgets и скрытыми seen actions в Bonus A]`

## 2. Cost curve и выбранный метод

Zero-shot success равен `[ZERO-SHOT MEAN]`. С чужой инструкцией он равен `[WRONG-INSTRUCTION MEAN]`. Это показывает `[LANGUAGE CONCLUSION]`.

Наивный fine-tune даёт `[BASELINE 5]`, `[BASELINE 10]` и `[BASELINE 25]` success.

Выбранный метод даёт `[METHOD 5]`, `[METHOD 10]` и `[METHOD 25]` success. Он обучает `[METHOD PARAMETERS]` параметров за `[METHOD TRAINING COST]`.

Метод `[BEATS OR DOES NOT BEAT]` baseline на `[BUDGETS]`. На `[UNSUPPORTED BUDGETS]` преимущество отсутствует.

`[FIGURE: одна cost curve для baseline и выбранного метода, отдельные задачи, среднее и разброс по training seeds]`

`[TABLE: baseline, выбранный метод и один контроль; параметры, время, success при 5, 10 и 25 демо]`

## 3. Фейлы и Bonus A

В первом фейле `[OBSERVATION 1]`. Мы предполагаем `[CAUSE 1A]`, а не `[CAUSE 1B]`. Это разделит эксперимент `[INTERVENTION 1]`.

Во втором фейле `[OBSERVATION 2]`. Мы предполагаем `[CAUSE 2A]`, а не `[CAUSE 2B]`. Это разделит эксперимент `[INTERVENTION 2]`.

В третьем фейле `[OBSERVATION 3]`. Мы предполагаем `[CAUSE 3A]`, а не `[CAUSE 3B]`. Это разделит эксперимент `[INTERVENTION 3]`.

`[FIGURE: три полоски реальных кадров; instruction, task, checkpoint и момент ошибки]`

В Bonus A predictor учится на соседних visual tokens seen-видео без actions. Decoder связывает предсказанное изменение с actions только на разрешённых target-демонстрациях.

Bonus A даёт `[BONUS 5]`, `[BONUS 10]` и `[BONUS 25]` success против `[CONTROL VALUES]`.

`[FIGURE: Bonus A против контроля на cost curve; actions для настоящего, предсказанного, нулевого и переставленного transition]`

## 4. Что изменилось после экспериментов

Мы ожидали `[PREDICTION 1]`, но получили `[RESULT 1]`. В исходном рассуждении мы не учли `[WHY 1]`.

Мы ожидали `[PREDICTION 2]`, но получили `[RESULT 2]`. Поэтому мы изменили решение `[DECISION 2]`.

Мы отвергли `[REJECTED METHOD 1]` после `[MEASUREMENT 1]`, потому что `[MECHANISM 1]`.

Мы отвергли `[REJECTED METHOD 2]` после `[MEASUREMENT 2]`, потому что `[MECHANISM 2]`.

`[REJECTED METHOD 3, IF NEEDED]`.

Самым неожиданным наблюдением было `[OBSERVATION]`. Мы объясняем его через `[EXPLANATION]`. Это поддерживает `[DIAGNOSTIC]`. От альтернативы его пока отделяет только `[MISSING TEST]`.

При 5, 10 и 25 демо мы выбираем `[FINAL CHOICE]`. Решение изменится, если `[FLIP CONDITION]`.
