# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,386 (6.7% break within 60 days). Each test class (2024, 2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,008 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2024 | 26023 | 2057 | 0.079 | 0.2824 | 0.6562 | 0.136 | 0.4352 |
| logistic | same as at-commit | class_2024 | 26023 | 2057 | 0.079 | 0.0575 | 0.6701 | 0.0715 | 0.2676 |
| boosted | same as at-commit | class_2024 | 26023 | 2057 | 0.079 | 0.0595 | 0.718 | 0.0697 | 0.2574 |
| logistic | all snapshots | class_2024 | 37685 | 2057 | 0.0546 | 0.0576 | 0.655 | 0.0506 | 0.2034 |
| boosted | all snapshots | class_2024 | 37685 | 2057 | 0.0546 | 0.0559 | 0.7363 | 0.0495 | 0.1934 |
| at_commit | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.3045 | 0.5998 | 0.1513 | 0.4797 |
| logistic | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.0593 | 0.659 | 0.0765 | 0.2838 |
| boosted | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.0607 | 0.7129 | 0.0746 | 0.2736 |
| logistic | all snapshots | class_2025 | 35259 | 2709 | 0.0768 | 0.0583 | 0.6669 | 0.0691 | 0.2602 |
| boosted | all snapshots | class_2025 | 35259 | 2709 | 0.0768 | 0.0581 | 0.7262 | 0.0673 | 0.2497 |
| at_commit | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.3095 | 0.6035 | 0.1548 | 0.478 |
| logistic | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.0686 | 0.6524 | 0.0812 | 0.2971 |
| boosted | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.0704 | 0.7452 | 0.0769 | 0.275 |
| logistic | all snapshots | class_2026 | 35000 | 3109 | 0.0888 | 0.0691 | 0.6459 | 0.0787 | 0.2898 |
| boosted | all snapshots | class_2026 | 35000 | 3109 | 0.0888 | 0.0698 | 0.747 | 0.0744 | 0.2672 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0851 | 0.615 | 0.3202 | 1.0239 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.1062 | 0.6094 | 0.3067 | 0.961 |
| logistic | all snapshots | class_2027 | 12990 | 656 | 0.0505 | 0.0634 | 0.6956 | 0.0468 | 0.1892 |
| boosted | all snapshots | class_2027 | 12990 | 656 | 0.0505 | 0.0567 | 0.8029 | 0.0435 | 0.1702 |
| logistic | training data | train_2022_2023 | 60111 | 3612 | 0.0601 | 0.0591 | 0.7068 | 0.0542 | 0.2115 |
| boosted | training data | train_2022_2023 | 60111 | 3612 | 0.0601 | 0.0678 | 0.8999 | 0.044 | 0.1657 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2024 | 30 | 0.216 | 0.0759 | 2.84 |
| logistic | same as at-commit | 2024 | 30 | 0.228 | 0.0759 | 3.0 |
| boosted | same as at-commit | 2024 | 30 | 0.2293 | 0.0759 | 3.02 |
| logistic | all snapshots | 2024 | 31 | 0.1987 | 0.0576 | 3.45 |
| boosted | all snapshots | 2024 | 31 | 0.1987 | 0.0576 | 3.45 |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0891 | 2.3 |
| logistic | same as at-commit | 2025 | 35 | 0.2457 | 0.0891 | 2.76 |
| boosted | same as at-commit | 2025 | 35 | 0.288 | 0.0891 | 3.23 |
| logistic | all snapshots | 2025 | 35 | 0.2366 | 0.0782 | 3.02 |
| boosted | all snapshots | 2025 | 35 | 0.2731 | 0.0782 | 3.49 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.2478 | 0.089 | 2.78 |
| boosted | same as at-commit | 2026 | 36 | 0.3333 | 0.089 | 3.75 |
| logistic | all snapshots | 2026 | 36 | 0.2367 | 0.0824 | 2.87 |
| boosted | all snapshots | 2026 | 36 | 0.31 | 0.0824 | 3.76 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.1981 | 0.0834 | 2.37 |
| boosted | all snapshots | 2027 | 21 | 0.2648 | 0.0834 | 3.17 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 3787 | 0.0158 | 0.0172 |
| logistic | 2-5% | 51216 | 0.0369 | 0.042 |
| logistic | 5-10% | 54439 | 0.068 | 0.0775 |
| logistic | 10-20% | 9738 | 0.1295 | 0.1632 |
| logistic | 20-35% | 1298 | 0.2531 | 0.2558 |
| logistic | 35%+ | 456 | 0.493 | 0.3772 |
| boosted | 0-2% | 18293 | 0.0127 | 0.0146 |
| boosted | 2-5% | 49843 | 0.0342 | 0.0342 |
| boosted | 5-10% | 33181 | 0.0706 | 0.0832 |
| boosted | 10-20% | 16617 | 0.1323 | 0.1691 |
| boosted | 20-35% | 2443 | 0.2438 | 0.2894 |
| boosted | 35%+ | 557 | 0.4792 | 0.5027 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7673 | 0.0684 | 0.1048 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12259 | 0.0394 | 0.0291 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4302 | 0.0899 | 0.1472 | 0.1362 |
| 2025 | 2026 | 30-60 days out | 4330 | 0.12 | 0.1539 | 0.167 |
| 2025 | 2026 | 0-30 days out | 4855 | 0.0986 | 0.1173 | 0.1291 |
| 2025 | 2026 | after early signing day | 1581 | 0.0324 | 0.1224 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.049 | 0.0508 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0393 | 0.0395 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0689 | 0.1054 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.1036 | 0.145 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5285 | 0.0868 | 0.1146 | 0.1048 |
| 2026 | 2025 | after early signing day | 3108 | 0.0196 | 0.0453 | 0.0788 |

## Coverage blend (fit on one held-out class, checked on the other)

next 60 days: blend not used (no improvement); before signing: blend not used (no improvement)

| target | blend | fit_on | checked_on | auc | top_25_hit_rate | predicted | actual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 60 days | False | 2025 | 2026 | 0.7472 | 0.31 | 0.0921 | 0.0888 |
| next 60 days | False | 2026 | 2025 | 0.7409 | 0.2731 | 0.0746 | 0.0768 |
| next 60 days | True | 2025 | 2026 | 0.745 | 0.31 | 0.0932 | 0.0888 |
| next 60 days | True | 2026 | 2025 | 0.7409 | 0.2823 | 0.0742 | 0.0768 |
| before signing | False | 2025 | 2026 | 0.7165 | 0.569 | 0.24 | 0.2392 |
| before signing | False | 2026 | 2025 | 0.7475 | 0.5505 | 0.2062 | 0.2043 |
| before signing | True | 2025 | 2026 | 0.7141 | 0.539 | 0.2422 | 0.2392 |
| before signing | True | 2026 | 2025 | 0.7477 | 0.5768 | 0.2059 | 0.2043 |

## Before signing: breaks at any point before the signing period ends

Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of 6 models trained on resampled commitments.

| checked_on | stage | snapshots | auc | calibrated | actual | top_25_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | all | 38308 | 0.7082 | 0.24 | 0.2392 | 0.569 |
| 2026 | 90-180 days out | 12344 | 0.6253 | 0.2363 | 0.25 |  |
| 2026 | 60-90 days out | 4373 | 0.6427 | 0.2248 | 0.2344 |  |
| 2026 | 30-60 days out | 4431 | 0.6723 | 0.2103 | 0.209 |  |
| 2026 | 0-30 days out | 7032 | 0.7349 | 0.1163 | 0.1303 |  |
| 2026 | 180+ days out | 7771 | 0.6788 | 0.4186 | 0.3864 |  |
| 2026 | after early signing day | 2357 | 0.8752 | 0.123 | 0.0878 |  |
| 2025 | all | 38716 | 0.7373 | 0.2062 | 0.2043 | 0.5505 |
| 2025 | 180+ days out | 6805 | 0.735 | 0.3506 | 0.3669 |  |
| 2025 | 90-180 days out | 11527 | 0.6799 | 0.2327 | 0.2167 |  |
| 2025 | 60-90 days out | 4348 | 0.6863 | 0.2158 | 0.204 |  |
| 2025 | 30-60 days out | 4487 | 0.6901 | 0.1909 | 0.1917 |  |
| 2025 | 0-30 days out | 7189 | 0.7119 | 0.1286 | 0.1145 |  |
| 2025 | after early signing day | 4360 | 0.8144 | 0.0444 | 0.0791 |  |

