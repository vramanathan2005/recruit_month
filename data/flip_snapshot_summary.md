# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,386 (6.7% break within 60 days). Each test class (2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,017 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.3045 | 0.5998 | 0.1513 | 0.4797 |
| logistic | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.0592 | 0.659 | 0.0765 | 0.2838 |
| boosted | same as at-commit | class_2025 | 31700 | 2709 | 0.0855 | 0.0614 | 0.713 | 0.0745 | 0.2735 |
| logistic | all snapshots | class_2025 | 35259 | 2709 | 0.0768 | 0.0582 | 0.6668 | 0.0691 | 0.2602 |
| boosted | all snapshots | class_2025 | 35259 | 2709 | 0.0768 | 0.0588 | 0.7263 | 0.0673 | 0.2497 |
| at_commit | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.3095 | 0.6035 | 0.1548 | 0.478 |
| logistic | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.0686 | 0.6509 | 0.0813 | 0.2975 |
| boosted | same as at-commit | class_2026 | 33769 | 3109 | 0.0921 | 0.0701 | 0.7461 | 0.0769 | 0.2752 |
| logistic | all snapshots | class_2026 | 35000 | 3109 | 0.0888 | 0.0691 | 0.6444 | 0.0788 | 0.2902 |
| boosted | all snapshots | class_2026 | 35000 | 3109 | 0.0888 | 0.0695 | 0.7479 | 0.0744 | 0.2675 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0851 | 0.6148 | 0.3202 | 1.0234 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.1057 | 0.6088 | 0.307 | 0.9636 |
| logistic | all snapshots | class_2027 | 12990 | 656 | 0.0505 | 0.0635 | 0.6955 | 0.0468 | 0.1893 |
| boosted | all snapshots | class_2027 | 12990 | 656 | 0.0505 | 0.0561 | 0.8027 | 0.0435 | 0.17 |
| logistic | training data | train_2022_2024 | 97796 | 5669 | 0.058 | 0.0574 | 0.6968 | 0.0526 | 0.2073 |
| boosted | training data | train_2022_2024 | 97796 | 5669 | 0.058 | 0.0547 | 0.8945 | 0.0463 | 0.1677 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0891 | 2.3 |
| logistic | same as at-commit | 2025 | 35 | 0.2469 | 0.0891 | 2.77 |
| boosted | same as at-commit | 2025 | 35 | 0.2846 | 0.0891 | 3.19 |
| logistic | all snapshots | 2025 | 35 | 0.2389 | 0.0782 | 3.05 |
| boosted | all snapshots | 2025 | 35 | 0.2686 | 0.0782 | 3.43 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.2433 | 0.089 | 2.73 |
| boosted | same as at-commit | 2026 | 36 | 0.3289 | 0.089 | 3.7 |
| logistic | all snapshots | 2026 | 36 | 0.2322 | 0.0824 | 2.82 |
| boosted | all snapshots | 2026 | 36 | 0.3022 | 0.0824 | 3.67 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.1962 | 0.0834 | 2.35 |
| boosted | all snapshots | 2027 | 21 | 0.2724 | 0.0834 | 3.27 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2779 | 0.0159 | 0.018 |
| logistic | 2-5% | 33738 | 0.0368 | 0.0462 |
| logistic | 5-10% | 37799 | 0.0684 | 0.0842 |
| logistic | 10-20% | 7529 | 0.13 | 0.1678 |
| logistic | 20-35% | 1027 | 0.2525 | 0.2658 |
| logistic | 35%+ | 377 | 0.494 | 0.3846 |
| boosted | 0-2% | 13445 | 0.0127 | 0.017 |
| boosted | 2-5% | 31570 | 0.0341 | 0.0365 |
| boosted | 5-10% | 23135 | 0.071 | 0.087 |
| boosted | 10-20% | 12917 | 0.1331 | 0.1767 |
| boosted | 20-35% | 1818 | 0.2439 | 0.3108 |
| boosted | 35%+ | 364 | 0.4978 | 0.6401 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7673 | 0.0684 | 0.1039 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12259 | 0.039 | 0.0283 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4302 | 0.0898 | 0.1453 | 0.1362 |
| 2025 | 2026 | 30-60 days out | 4330 | 0.1193 | 0.1511 | 0.167 |
| 2025 | 2026 | 0-30 days out | 4855 | 0.0981 | 0.1155 | 0.1291 |
| 2025 | 2026 | after early signing day | 1581 | 0.0321 | 0.1185 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0495 | 0.0517 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0398 | 0.0406 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0697 | 0.1085 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.1048 | 0.1481 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5285 | 0.0878 | 0.1164 | 0.1048 |
| 2026 | 2025 | after early signing day | 3108 | 0.0199 | 0.0465 | 0.0788 |

## Coverage blend (fit on one held-out class, checked on the other)

next 60 days: blend not used (no improvement); before signing: blend not used (no improvement)

| target | blend | fit_on | checked_on | auc | top_25_hit_rate | predicted | actual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 60 days | False | 2025 | 2026 | 0.7477 | 0.3022 | 0.0906 | 0.0888 |
| next 60 days | False | 2026 | 2025 | 0.7409 | 0.2686 | 0.0762 | 0.0768 |
| next 60 days | True | 2025 | 2026 | 0.7456 | 0.3122 | 0.0916 | 0.0888 |
| next 60 days | True | 2026 | 2025 | 0.741 | 0.2777 | 0.0758 | 0.0768 |
| before signing | False | 2025 | 2026 | 0.718 | 0.584 | 0.2383 | 0.2392 |
| before signing | False | 2026 | 2025 | 0.7473 | 0.5316 | 0.2065 | 0.2043 |
| before signing | True | 2025 | 2026 | 0.7154 | 0.543 | 0.2409 | 0.2392 |
| before signing | True | 2026 | 2025 | 0.7475 | 0.5632 | 0.2062 | 0.2043 |

## Before signing: breaks at any point before the signing period ends

Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of 6 models trained on resampled commitments.

| checked_on | stage | snapshots | auc | calibrated | actual | top_25_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | all | 38308 | 0.709 | 0.2383 | 0.2392 | 0.584 |
| 2026 | 90-180 days out | 12344 | 0.6269 | 0.2348 | 0.25 |  |
| 2026 | 60-90 days out | 4373 | 0.645 | 0.2238 | 0.2344 |  |
| 2026 | 30-60 days out | 4431 | 0.6735 | 0.2096 | 0.209 |  |
| 2026 | 0-30 days out | 7032 | 0.734 | 0.1162 | 0.1303 |  |
| 2026 | 180+ days out | 7771 | 0.6782 | 0.4146 | 0.3864 |  |
| 2026 | after early signing day | 2357 | 0.8768 | 0.121 | 0.0878 |  |
| 2025 | all | 38716 | 0.7371 | 0.2065 | 0.2043 | 0.5316 |
| 2025 | 180+ days out | 6805 | 0.7381 | 0.3507 | 0.3669 |  |
| 2025 | 90-180 days out | 11527 | 0.6781 | 0.2334 | 0.2167 |  |
| 2025 | 60-90 days out | 4348 | 0.6834 | 0.2161 | 0.204 |  |
| 2025 | 30-60 days out | 4487 | 0.6858 | 0.1911 | 0.1917 |  |
| 2025 | 0-30 days out | 7189 | 0.7111 | 0.1284 | 0.1145 |  |
| 2025 | after early signing day | 4360 | 0.8123 | 0.0449 | 0.0791 |  |

