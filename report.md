# Адаптация SmolVLA к новым задачам LIBERO

## 1. Постановка, данные и предсказания

Мы проверяем, сколько target-демонстраций нужно SmolVLA после претренинга на `libero_90`.

Seen-претрен использует 5000 шагов на 450 равномерно выбранных эпизодах из 4500 эпизодов `libero_90`. Target содержит первые 5, 10 или 25 демонстраций каждой из первых трёх задач `libero_goal`.

Каждая финальная точка содержит 20 роллаутов. Мы используем training seeds 0 и 1 и одинаковый eval seed 1000.

До запусков мы ожидали `[PREDICTION ABOUT ZERO-SHOT]`.

Для подмешивания seen мы ожидали `[PREDICTION ABOUT MIX CURVE]`.

Для LoRA мы ожидали `[PREDICTION ABOUT LORA CURVE]`.

Мы ожидали, что video representation сильнее всего поможет при 5 демо. К 25 демо наивный fine-tune должен частично сократить разрыв.

`[FIGURE: split с suite, task IDs, episode budgets и скрытыми seen actions в Bonus A]`

## 2. Cost curve и выбранный метод

Zero-shot success равен 0/60. С чужой инструкцией он также равен 0/60. Этот контроль не отделяет влияние языка, потому что исходная политика не решает target-задачи.

Подмешивание seen даёт 0.742, 0.817 и 0.917 success.

LoRA даёт 0.575, 0.717 и 0.858 success. Она обучает 1.49 млн adapter-параметров.

Мы выбираем подмешивание seen. Оно лучше LoRA на 0.167, 0.100 и 0.058 при 5, 10 и 25 демонстрациях.

Эти числа усреднены по трём задачам и двум training seeds. На task 2 при пяти демонстрациях LoRA лучше mix на 0.075, поэтому преимущество mix не одинаково для каждой задачи.

`[FIGURE: одна cost curve для mix и LoRA, отдельные задачи, среднее и разброс по training seeds]`

`[TABLE: mix, LoRA и zero-shot; параметры, время, success при 5, 10 и 25 демо]`

## 3. Фейлы и Bonus A

В первом фейле `[OBSERVATION 1]`. Мы предполагаем `[CAUSE 1A]`, а не `[CAUSE 1B]`. Это разделит эксперимент `[INTERVENTION 1]`.

Во втором фейле `[OBSERVATION 2]`. Мы предполагаем `[CAUSE 2A]`, а не `[CAUSE 2B]`. Это разделит эксперимент `[INTERVENTION 2]`.

В третьем фейле `[OBSERVATION 3]`. Мы предполагаем `[CAUSE 3A]`, а не `[CAUSE 3B]`. Это разделит эксперимент `[INTERVENTION 3]`.

`[FIGURE: три полоски реальных кадров; instruction, task, checkpoint и момент ошибки]`

В Bonus A predictor учится на соседних visual tokens seen-видео без actions. Decoder связывает предсказанное изменение с actions только на разрешённых target-демонстрациях.

Bonus A даёт 0/60 success при пяти target-демонстрациях против 0.742 у mix и 0.575 у LoRA. Внутренние representation и policy losses снизились, но это не перенеслось в поведение робота.

`[FIGURE: Bonus A против контроля на cost curve; actions для настоящего, предсказанного, нулевого и переставленного transition]`

## 4. Что изменилось после экспериментов

Мы ожидали `[PREDICTION 1]`, но получили `[RESULT 1]`. В исходном рассуждении мы не учли `[WHY 1]`.

Мы ожидали `[PREDICTION 2]`, но получили `[RESULT 2]`. Поэтому мы изменили решение `[DECISION 2]`.

Мы отвергли `[REJECTED METHOD 1]` после `[MEASUREMENT 1]`, потому что `[MECHANISM 1]`.

Мы отвергли `[REJECTED METHOD 2]` после `[MEASUREMENT 2]`, потому что `[MECHANISM 2]`.

`[REJECTED METHOD 3, IF NEEDED]`.

Самым неожиданным наблюдением было `[OBSERVATION]`. Мы объясняем его через `[EXPLANATION]`. Это поддерживает `[DIAGNOSTIC]`. От альтернативы его пока отделяет только `[MISSING TEST]`.

При 5, 10 и 25 демо мы выбираем `[FINAL CHOICE]`. Решение изменится, если `[FLIP CONDITION]`.
