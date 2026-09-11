| course | `--speed` | polyline | spline arc | ref speed at gates (min/median/max) | peak ref speed | nominal / median |
|---|---|---|---|---|---|---|
| 60° | 10.164 m/s | 38.40 m | 39.40 m | 3.81 / 5.51 / 6.49 m/s | 28.9 m/s | 1.84x |
| 90° | 8.367 m/s | 38.40 m | 39.95 m | 3.09 / 3.90 / 5.39 m/s | 28.7 m/s | 2.14x |
| 120° | 6.648 m/s | 38.40 m | 40.01 m | 2.07 / 2.31 / 4.32 m/s | 26.7 m/s | 2.88x |

Decomposition of that last column into its two independent causes:

| course | total_time | mean = arc/total_time | startup accounting | parameterisation | product | reported |
|---|---|---|---|---|---|---|
| 60° | 6.778 s | 5.812 m/s | 1.75x | 1.05x | 1.84x | 1.84x |
| 90° | 7.589 s | 5.264 m/s | 1.59x | 1.35x | 2.14x | 2.14x |
| 120° | 8.776 s | 4.559 m/s | 1.46x | 1.97x | 2.88x | 2.88x |
