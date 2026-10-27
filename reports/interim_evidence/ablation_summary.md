# Daily-Stride Transformer Ablations

| model | pinball_loss | crps | coverage_90 | median_mae | n_observations |
|---|---|---|---|---|---|
| hf_transformer_daily_embed_observed_weather_oracle | 0.5653 | 1.0536 | 0.949 | 1.5108 | 435600 |
| hf_transformer_daily_embed_noweather | 0.5755 | 1.0727 | 0.939 | 1.5426 | 435600 |
| hf_transformer_daily_noembed_noweather | 0.5770 | 1.0753 | 0.949 | 1.5419 | 435600 |

All rows use the common 435,600-row 2025 validation window. `observed_weather_oracle` uses realized future weather and is not deployable; no model is described as using forecast weather.
