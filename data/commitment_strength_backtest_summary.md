# Commitment Strength Backtest

- Commitment events tested: 18027
- Known outcomes: 12883
- Decommit/flip outcomes: 3196
- Overall decommit/flip rate: 24.8%

| Outcome | Count |
| --- | ---: |
| stuck | 9687 |
| open | 5144 |
| decommitted | 2943 |
| flipped | 253 |

| Score Bucket | Known Events | Decommit/Flip | Decommit/Flip Rate | Stick Rate |
| --- | ---: | ---: | ---: | ---: |
| Weak / Attack | 845 | 481 | 56.9% | 43.1% |
| Vulnerable | 5452 | 1878 | 34.4% | 65.5% |
| Stable / Monitor | 3425 | 616 | 18.0% | 82.0% |
| Strong | 3161 | 221 | 7.0% | 93.0% |

This backtest scores each historical commitment event using pre-outcome signals only:
timing, prior commitment/decommit history, pre-commit offers/visits, rating tier, distance, and class crowding.
It does not use post-commit events as an input, because those are the outcome being tested.
