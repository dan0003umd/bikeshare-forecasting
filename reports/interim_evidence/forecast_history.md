# Forecast-Origin History Available to Each Model

| Model | Fitted parameters/history | Context available at a 2025 local-midnight origin | Earlier 2025 observed demand? |
|---|---|---|---|
| Seasonal naive, rolling 8-week | No fitted parameters; empirical 19 quantiles | The previous eight observations for the same station, weekday, and local hour | Yes, when those observations precede the origin |
| Seasonal naive, frozen training | No fitted parameters; empirical 19 quantiles frozen at 2024 cutoff | The last eight training-period observations for the same station, weekday, and local hour | No |
| Per-station Negative Binomial | GLM fit on 2023--2024 only | Calendar covariates only; no demand lag/context | No |
| HF Transformer, first run | Parameters fit on 2023--2024 weekly-stride windows | 336 prior hourly values (168 context + maximum 168-hour lag) and local calendar features | Yes, where those prior values precede the origin |
| HF Transformer, daily stride + embeddings | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values, local calendar features, and station embedding | Yes, where those prior values precede the origin |
| HF Transformer, daily stride without embeddings | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values and local calendar features; no station embedding | Yes, where those prior values precede the origin |
| HF Transformer, observed-weather ORACLE | Parameters fit on 2023--2024 daily-stride windows | 336 prior hourly values, station embedding, local calendar features, and realized future weather | Yes, where those prior values precede the origin |

All models are scored on the same 24 future hours for each of 363 eligible local days. The rolling seasonal baseline and transformer are causal with respect to an origin, but they are not frozen-history comparators; both can consume demand observed earlier in 2025 during sequential validation.
