| course | `--speed` | polyline | spline arc | ref speed at gates (min/median/max) | peak ref speed | nominal / median |
|---|---|---|---|---|---|---|
| 60° | 10.398 m/s | 38.40 m | 39.40 m | 3.84 / 5.60 / 6.60 m/s | 29.4 m/s | 1.86x |
| 90° | 8.250 m/s | 38.40 m | 39.95 m | 3.08 / 3.86 / 5.34 m/s | 28.4 m/s | 2.14x |
| 120° | 7.117 m/s | 38.40 m | 40.01 m | 2.11 / 2.44 / 4.55 m/s | 28.2 m/s | 2.92x |

Decomposition of that last column into its two independent causes:

| course | total_time | mean = arc/total_time | startup accounting | parameterisation | product | reported |
|---|---|---|---|---|---|---|
| 60° | 6.693 s | 5.886 m/s | 1.77x | 1.05x | 1.86x | 1.86x |
| 90° | 7.655 s | 5.220 m/s | 1.58x | 1.35x | 2.14x | 2.14x |
| 120° | 8.395 s | 4.766 m/s | 1.49x | 1.96x | 2.92x | 2.92x |
