# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,500 (6.7% break within 60 days). Each test class (2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,018 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.3044 | 0.5992 | 0.1514 | 0.48 |
| logistic | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0586 | 0.6545 | 0.077 | 0.2859 |
| boosted | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0622 | 0.7033 | 0.075 | 0.276 |
| logistic | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0577 | 0.6625 | 0.0695 | 0.2621 |
| boosted | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0596 | 0.717 | 0.0677 | 0.2519 |
| at_commit | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.3095 | 0.6035 | 0.1549 | 0.4781 |
| logistic | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0671 | 0.6425 | 0.0821 | 0.3004 |
| boosted | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0648 | 0.7324 | 0.0789 | 0.2823 |
| logistic | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0678 | 0.6347 | 0.0796 | 0.2931 |
| boosted | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0643 | 0.7338 | 0.0763 | 0.2743 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0772 | 0.6118 | 0.3276 | 1.0501 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0745 | 0.6076 | 0.3281 | 1.0817 |
| logistic | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.061 | 0.6841 | 0.0467 | 0.19 |
| boosted | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0395 | 0.8002 | 0.0446 | 0.1719 |
| logistic | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0578 | 0.6837 | 0.0531 | 0.2095 |
| boosted | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0551 | 0.8927 | 0.0467 | 0.1694 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0892 | 2.29 |
| logistic | same as at-commit | 2025 | 35 | 0.2514 | 0.0892 | 2.82 |
| boosted | same as at-commit | 2025 | 35 | 0.2731 | 0.0892 | 3.06 |
| logistic | all snapshots | 2025 | 35 | 0.2343 | 0.0783 | 2.99 |
| boosted | all snapshots | 2025 | 35 | 0.2503 | 0.0783 | 3.2 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.24 | 0.089 | 2.7 |
| boosted | same as at-commit | 2026 | 36 | 0.3089 | 0.089 | 3.47 |
| logistic | all snapshots | 2026 | 36 | 0.21 | 0.0825 | 2.55 |
| boosted | all snapshots | 2026 | 36 | 0.2667 | 0.0825 | 3.23 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.2019 | 0.0834 | 2.42 |
| boosted | all snapshots | 2027 | 21 | 0.2648 | 0.0834 | 3.17 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2367 | 0.0162 | 0.0173 |
| logistic | 2-5% | 33458 | 0.0371 | 0.0458 |
| logistic | 5-10% | 38870 | 0.0686 | 0.0878 |
| logistic | 10-20% | 7548 | 0.1276 | 0.1611 |
| logistic | 20-35% | 850 | 0.2488 | 0.2506 |
| logistic | 35%+ | 187 | 0.4394 | 0.3636 |
| boosted | 0-2% | 16371 | 0.0122 | 0.0173 |
| boosted | 2-5% | 30663 | 0.0338 | 0.0409 |
| boosted | 5-10% | 22270 | 0.0725 | 0.0964 |
| boosted | 10-20% | 12762 | 0.1316 | 0.1836 |
| boosted | 20-35% | 1059 | 0.2462 | 0.3305 |
| boosted | 35%+ | 155 | 0.4908 | 0.6774 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7674 | 0.0634 | 0.091 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12265 | 0.0381 | 0.0278 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4304 | 0.0817 | 0.1273 | 0.1364 |
| 2025 | 2026 | 30-60 days out | 4332 | 0.1067 | 0.135 | 0.1674 |
| 2025 | 2026 | 0-30 days out | 4857 | 0.0896 | 0.1076 | 0.1295 |
| 2025 | 2026 | after early signing day | 1582 | 0.0313 | 0.1133 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0518 | 0.0587 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0395 | 0.041 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0722 | 0.1208 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.1059 | 0.1661 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5288 | 0.0876 | 0.1268 | 0.105 |
| 2026 | 2025 | after early signing day | 3119 | 0.0199 | 0.0479 | 0.0792 |

