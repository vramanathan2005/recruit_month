# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,500 (6.7% break within 60 days). Each test class (2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,018 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.3044 | 0.5992 | 0.1514 | 0.48 |
| logistic | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0611 | 0.6548 | 0.0766 | 0.2844 |
| boosted | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0626 | 0.7087 | 0.0748 | 0.2747 |
| logistic | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.06 | 0.6632 | 0.0692 | 0.2608 |
| boosted | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0597 | 0.7254 | 0.0675 | 0.2504 |
| at_commit | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.3095 | 0.6035 | 0.1549 | 0.4781 |
| logistic | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0691 | 0.6538 | 0.0813 | 0.2972 |
| boosted | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0669 | 0.7471 | 0.0781 | 0.278 |
| logistic | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0696 | 0.6474 | 0.0788 | 0.2899 |
| boosted | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0665 | 0.7481 | 0.0755 | 0.2703 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0856 | 0.6163 | 0.3195 | 1.0207 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.1158 | 0.6214 | 0.299 | 0.9289 |
| logistic | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0642 | 0.6943 | 0.0469 | 0.1898 |
| boosted | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0624 | 0.8013 | 0.0435 | 0.1711 |
| logistic | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0576 | 0.6963 | 0.0528 | 0.2082 |
| boosted | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0637 | 0.8946 | 0.0454 | 0.1693 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0892 | 2.29 |
| logistic | same as at-commit | 2025 | 35 | 0.2549 | 0.0892 | 2.86 |
| boosted | same as at-commit | 2025 | 35 | 0.2686 | 0.0892 | 3.01 |
| logistic | all snapshots | 2025 | 35 | 0.2457 | 0.0783 | 3.14 |
| boosted | all snapshots | 2025 | 35 | 0.2571 | 0.0783 | 3.28 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.2533 | 0.089 | 2.85 |
| boosted | same as at-commit | 2026 | 36 | 0.3356 | 0.089 | 3.77 |
| logistic | all snapshots | 2026 | 36 | 0.2433 | 0.0825 | 2.95 |
| boosted | all snapshots | 2026 | 36 | 0.3111 | 0.0825 | 3.77 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.2 | 0.0834 | 2.4 |
| boosted | all snapshots | 2027 | 21 | 0.261 | 0.0834 | 3.13 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2374 | 0.016 | 0.0219 |
| logistic | 2-5% | 33340 | 0.0371 | 0.0461 |
| logistic | 5-10% | 38337 | 0.0683 | 0.0826 |
| logistic | 10-20% | 7711 | 0.1302 | 0.1659 |
| logistic | 20-35% | 1094 | 0.2504 | 0.2605 |
| logistic | 35%+ | 424 | 0.5034 | 0.3797 |
| boosted | 0-2% | 10897 | 0.0129 | 0.0176 |
| boosted | 2-5% | 30000 | 0.0348 | 0.0327 |
| boosted | 5-10% | 28115 | 0.0703 | 0.0832 |
| boosted | 10-20% | 12750 | 0.1291 | 0.1845 |
| boosted | 20-35% | 1198 | 0.2403 | 0.379 |
| boosted | 35%+ | 320 | 0.4632 | 0.5125 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7674 | 0.0675 | 0.0889 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12265 | 0.0407 | 0.0254 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4304 | 0.0849 | 0.1421 | 0.1364 |
| 2025 | 2026 | 30-60 days out | 4332 | 0.1053 | 0.1519 | 0.1674 |
| 2025 | 2026 | 0-30 days out | 4857 | 0.0894 | 0.1159 | 0.1295 |
| 2025 | 2026 | after early signing day | 1582 | 0.0351 | 0.1192 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0582 | 0.061 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0447 | 0.0437 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0693 | 0.1083 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.0944 | 0.1472 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5288 | 0.0804 | 0.1142 | 0.105 |
| 2026 | 2025 | after early signing day | 3119 | 0.0209 | 0.0423 | 0.0792 |

## Before signing: breaks at any point before the signing period ends

Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of 6 models trained on resampled commitments.

| checked_on | stage | snapshots | auc | calibrated | actual | top_25_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | all | 38325 | 0.7045 | 0.2381 | 0.2395 | 0.559 |
| 2026 | 90-180 days out | 12350 | 0.6187 | 0.2358 | 0.2504 |  |
| 2026 | 60-90 days out | 4375 | 0.6322 | 0.2267 | 0.2347 |  |
| 2026 | 30-60 days out | 4433 | 0.6635 | 0.2112 | 0.2093 |  |
| 2026 | 0-30 days out | 7036 | 0.7311 | 0.1168 | 0.1308 |  |
| 2026 | 180+ days out | 7772 | 0.6769 | 0.4045 | 0.3865 |  |
| 2026 | after early signing day | 2359 | 0.8789 | 0.1358 | 0.0877 |  |
| 2025 | all | 38734 | 0.7349 | 0.2055 | 0.2043 | 0.5821 |
| 2025 | 180+ days out | 6805 | 0.741 | 0.3563 | 0.3669 |  |
| 2025 | 90-180 days out | 11527 | 0.6755 | 0.2329 | 0.2167 |  |
| 2025 | 60-90 days out | 4348 | 0.6786 | 0.214 | 0.204 |  |
| 2025 | 30-60 days out | 4487 | 0.6803 | 0.1894 | 0.1917 |  |
| 2025 | 0-30 days out | 7192 | 0.6981 | 0.1279 | 0.1146 |  |
| 2025 | after early signing day | 4375 | 0.8019 | 0.0339 | 0.0795 |  |

