# 2025 event-day validation slices

This is a validation-only, post-hoc diagnostic: no model was retrained. Event labels came from `load_development_model_table()` (which filters at the local-2026 boundary) and the typed event calendar.

## Event-day counts

| category | calendar_2025_days | valid_evaluation_days | inference_note |
|---|---|---|---|
| nationals_home_game | 78 | 78 | descriptive slice; not causal inference |
| cherry_blossom_festival | 25 | 25 | descriptive slice; not causal inference |
| us_holiday | 11 | 11 | descriptive slice; not causal inference |
| marathon (not in event file) | 0 | 0 | not in event file |
| any_event | 102 | 102 | descriptive slice; not causal inference |

`any_event` is the union of Nationals home games, Cherry Blossom Festival days, and US holidays. It contains 102 of the 363 valid fixed-24-hour evaluation days. Categories with fewer than five days are marked **too few days for inference**; the metrics are descriptive only. Marathon is **not in the event file**, rather than a verified zero-event category.

12 flagged dates have more than one category. Every date is included exactly once in the aggregate `any_event` slice; see `event_dates_2025.csv` for the underlying flags.

## Seasonal composition by month

Counts use the fixed 363-day evaluation window; `calendar_days` is the unfiltered calendar length.

| month | calendar_days | valid_evaluation_days | event_days | non_event_days | nationals_home_game | cherry_blossom_festival | us_holiday | marathon |
|---|---|---|---|---|---|---|---|---|
| 2025-01 | 31 | 31 | 2 | 29 | 0 | 0 | 2 | 0 |
| 2025-02 | 28 | 28 | 1 | 27 | 0 | 0 | 1 | 0 |
| 2025-03 | 31 | 30 | 12 | 18 | 3 | 12 | 0 | 0 |
| 2025-04 | 30 | 30 | 20 | 10 | 13 | 13 | 0 | 0 |
| 2025-05 | 31 | 31 | 11 | 20 | 10 | 0 | 1 | 0 |
| 2025-06 | 30 | 30 | 13 | 17 | 13 | 0 | 1 | 0 |
| 2025-07 | 31 | 31 | 11 | 20 | 11 | 0 | 1 | 0 |
| 2025-08 | 31 | 31 | 16 | 15 | 16 | 0 | 0 | 0 |
| 2025-09 | 30 | 30 | 12 | 18 | 12 | 0 | 1 | 0 |
| 2025-10 | 31 | 31 | 1 | 30 | 0 | 0 | 1 | 0 |
| 2025-11 | 30 | 29 | 2 | 27 | 0 | 0 | 2 | 0 |
| 2025-12 | 31 | 31 | 1 | 30 | 0 | 0 | 1 | 0 |

## Model slices

Each model has exactly 435,600 rows when its event and non-event rows are combined. Pinball is averaged over the shared 19 quantiles; coverage is the q05--q95 interval.

| model | slice | pinball_loss | mean_actual_demand | relative_pinball | median_mae | coverage_90 | n_observations | days |
|---|---|---|---|---|---|---|---|---|
| seasonal_naive_rolling_8week | event | 0.745476078189232 | 3.6037009803921567 | 0.20686402180574617 | 1.9226838235294117 | 0.764640522875817 | 122400 | 102 |
| seasonal_naive_rolling_8week | non_event | 0.6347719356453335 | 3.0354246487867176 | 0.20912129573009058 | 1.655499680715198 | 0.7906641123882503 | 313200 | 261 |
| seasonal_naive_frozen_train_8week | event | 0.8508180984694731 | 3.6037009803921567 | 0.23609564253493823 | 2.1677124183006535 | 0.7263643790849673 | 122400 | 102 |
| seasonal_naive_frozen_train_8week | non_event | 0.7095300378741705 | 3.0354246487867176 | 0.2337498439164929 | 1.8361494252873563 | 0.7745945083014049 | 313200 | 261 |
| per_station_negative_binomial | event | 0.9046351049191608 | 3.6037009803921567 | 0.25102945828227896 | 2.3874754901960786 | 0.9570751633986928 | 122400 | 102 |
| per_station_negative_binomial | non_event | 0.7808988539356052 | 3.0354246487867176 | 0.2572618148329712 | 2.049007024265645 | 0.9617975734355044 | 313200 | 261 |
| hf_transformer_negative_binomial | event | 0.7015670149638803 | 3.6037009803921567 | 0.19467958600925192 | 1.8656127450980393 | 0.9283496732026144 | 122400 | 102 |
| hf_transformer_negative_binomial | non_event | 0.6083398547254151 | 3.0354246487867176 | 0.2004134264932497 | 1.6243997445721583 | 0.934198595146871 | 313200 | 261 |
| hf_transformer_daily_embed_noweather | event | 0.6403248495012039 | 3.6037009803921567 | 0.177685344312758 | 1.7142933006535948 | 0.932312091503268 | 122400 | 102 |
| hf_transformer_daily_embed_noweather | non_event | 0.5502002533272837 | 3.0354246487867176 | 0.18125973034685705 | 1.475552362707535 | 0.941433588761175 | 313200 | 261 |
| hf_transformer_daily_noembed_noweather | event | 0.6418731950894393 | 3.6037009803921567 | 0.1781149986033498 | 1.7137418300653595 | 0.9452777777777778 | 122400 | 102 |
| hf_transformer_daily_noembed_noweather | non_event | 0.5516944490656718 | 3.0354246487867176 | 0.18175198296758519 | 1.4747812899106003 | 0.9503639846743295 | 313200 | 261 |
| hf_transformer_daily_embed_observed_weather_oracle | event | 0.6297180157808737 | 3.6037009803921567 | 0.17474202748984669 | 1.6832434640522875 | 0.9447875816993464 | 122400 | 102 |
| hf_transformer_daily_embed_observed_weather_oracle | non_event | 0.5401825124353028 | 3.0354246487867176 | 0.17795945376249675 | 1.4434355044699871 | 0.9502362707535121 | 313200 | 261 |
| xgboost_shared_quantile | event | 0.6207292122292184 | 3.6037009803921567 | 0.172247701906075 | 1.6658682781211165 | 0.8972712418300653 | 122400 | 102 |
| xgboost_shared_quantile | non_event | 0.5423255220468707 | 3.0354246487867176 | 0.17866545369974587 | 1.455755224401951 | 0.9010121328224776 | 313200 | 261 |
| transformer_xgbfeatures_longcontext_nb | event | 0.6533733230134159 | 3.603700876235962 | 0.18130620310969298 | 1.7499264705882354 | 0.939983660130719 | 122400 | 102 |
| transformer_xgbfeatures_longcontext_nb | non_event | 0.5571468289977819 | 3.0354247093200684 | 0.1835482287823183 | 1.4957439335887612 | 0.9469667943805875 | 313200 | 261 |
| transformer_xgbfeatures_longcontext_quantile | event | 0.6666664475740638 | 3.603700876235962 | 0.1849949456044731 | 1.785849805118684 | 0.9082271241830066 | 122400 | 102 |
| transformer_xgbfeatures_longcontext_quantile | non_event | 0.5712998924096695 | 3.0354247093200684 | 0.18821085914453137 | 1.533834342684047 | 0.9136877394636015 | 313200 | 261 |

## Month-matched event comparison

For every month with event days, event and non-event scores are calculated separately and then averaged with weights equal to the month's event-day count. Relative pinball is pinball divided by mean actual demand on the same slice.

| model | event_months | event_days_weight | event_days | matched_non_event_days | event_pinball_loss | non_event_pinball_loss | event_minus_non_event_pinball_loss | event_relative_pinball | non_event_relative_pinball | event_minus_non_event_relative_pinball | event_median_mae | non_event_median_mae | event_minus_non_event_median_mae | event_coverage_90 | non_event_coverage_90 | event_minus_non_event_coverage_90 | event_mean_actual_demand | non_event_mean_actual_demand | event_minus_non_event_mean_actual_demand |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seasonal_naive_rolling_8week | 12 | 102 | 102 | 261 | 0.7454760781892319 | 0.6799686503602105 | 0.06550742782902141 | 0.22703214414886594 | 0.19229353179943842 | 0.03473861234942752 | 1.922683823529412 | 1.7667274638911108 | 0.15595635963830112 | 0.7646405228758171 | 0.7838672069220847 | -0.019226684046267595 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| seasonal_naive_frozen_train_8week | 12 | 102 | 102 | 261 | 0.8508180984694732 | 0.8096609630739482 | 0.04115713539552501 | 0.25027464083638923 | 0.22620609809093928 | 0.02406854274544995 | 2.167712418300653 | 2.0715485963498668 | 0.09616382195078632 | 0.7263643790849672 | 0.7423839811586369 | -0.016019602073669703 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| per_station_negative_binomial | 12 | 102 | 102 | 261 | 0.9046351049191607 | 0.9081817159274067 | -0.003546611008246048 | 0.2588414055009302 | 0.25368748912287264 | 0.005153916378057566 | 2.3874754901960786 | 2.392284039614393 | -0.004808549418314545 | 0.9570751633986927 | 0.9547315174792778 | 0.0023436459194149384 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| hf_transformer_negative_binomial | 12 | 102 | 102 | 261 | 0.7015670149638803 | 0.6910520641555101 | 0.010514950808370171 | 0.20001243847432448 | 0.19361249108283132 | 0.00639994739149316 | 1.8656127450980393 | 1.846908604541283 | 0.018704140556756288 | 0.9283496732026143 | 0.9291045321685882 | -0.0007548589659739369 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| hf_transformer_daily_embed_noweather | 12 | 102 | 102 | 261 | 0.6403248495012039 | 0.6138071079287546 | 0.02651774157244935 | 0.18232925934308494 | 0.17248147768592362 | 0.009847781657161325 | 1.714293300653595 | 1.6456854462689465 | 0.06860785438464845 | 0.9323120915032678 | 0.938463438299593 | -0.00615134679632523 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| hf_transformer_daily_noembed_noweather | 12 | 102 | 102 | 261 | 0.6418731950894393 | 0.6163629003725357 | 0.02551029471690358 | 0.18331889446301666 | 0.1732099259487922 | 0.010108968514224465 | 1.7137418300653593 | 1.646671560694433 | 0.0670702693709262 | 0.9452777777777778 | 0.9477165797134219 | -0.0024388019356441637 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| hf_transformer_daily_embed_observed_weather_oracle | 12 | 102 | 102 | 261 | 0.6297180157808738 | 0.6104551869544186 | 0.019262828826455247 | 0.18065127683894677 | 0.1710772993173592 | 0.009573977521587562 | 1.6832434640522875 | 1.6322919752871599 | 0.05095148876512767 | 0.9447875816993465 | 0.9480239192006907 | -0.003236337501344222 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| xgboost_shared_quantile | 12 | 102 | 102 | 261 | 0.6207292122292184 | 0.6105808358271818 | 0.010148376402036585 | 0.17682622941525497 | 0.17124587903579375 | 0.005580350379461224 | 1.6658682781211167 | 1.6377070087787755 | 0.0281612693423412 | 0.8972712418300655 | 0.8986335247875976 | -0.0013622829575321571 | 3.603700980392157 | 3.596896894639207 | 0.0068040857529503285 |
| transformer_xgbfeatures_longcontext_nb | 12 | 102 | 102 | 261 | 0.6533733230134158 | 0.6253256332687844 | 0.028047689744631343 | 0.18530141022379926 | 0.17566578512588463 | 0.00963562509791463 | 1.7499264705882351 | 1.6805168960091488 | 0.06940957457908636 | 0.9399836601307189 | 0.9453636206454634 | -0.005379960514744542 | 3.6037010083011554 | 3.596896868126065 | 0.006804140175090456 |
| transformer_xgbfeatures_longcontext_quantile | 12 | 102 | 102 | 261 | 0.6666664475740638 | 0.6389018051531737 | 0.02776464242089005 | 0.19007039256155947 | 0.1795172244503801 | 0.010553168111179373 | 1.7858498051186837 | 1.714993931283147 | 0.07085587383553671 | 0.9082271241830064 | 0.9113725828078206 | -0.0031454586248141903 | 3.6037010083011554 | 3.596896868126065 | 0.006804140175090456 |

## Paired XGBoost minus embedded daily transformer

Intervals use 10,000 seed-42 resamples of complete local days within each slice, rather than resampling rows. Negative differences favour XGBoost; an interval containing zero is not distinguishable from zero by this diagnostic.

| metric | slice | difference_xgboost_minus_daily_embed | ci95_low | ci95_high | ci_includes_zero | days | n_observations | bootstrap_draws | seed |
|---|---|---|---|---|---|---|---|---|---|
| pinball | event | -0.0195956372719856 | -0.031894829009789794 | -0.009185975701841292 | False | 102 | 122400 | 10000 | 42 |
| median_mae | event | -0.04842502253247816 | -0.08102492898415935 | -0.021035257717076376 | False | 102 | 122400 | 10000 | 42 |
| pinball | non_event | -0.007874731280413088 | -0.013956332286482515 | -0.0027023727738487613 | False | 261 | 313200 | 10000 | 42 |
| median_mae | non_event | -0.01979713830558392 | -0.03584668366718408 | -0.006050811460458879 | False | 261 | 313200 | 10000 | 42 |

## Month-matched paired XGBoost minus embedded daily transformer

Intervals use 10,000 seed-42 day resamples stratified within event-containing months, with the same event-day weights. Negative differences favour XGBoost.

| metric | slice | difference_xgboost_minus_daily_embed | ci95_low | ci95_high | ci_includes_zero | event_months | event_days_weight | days | n_observations | bootstrap_draws | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pinball | event | -0.0195956372719856 | -0.03115023226949316 | -0.010080126835865268 | False | 12 | 102 | 102 | 122400 | 10000 | 42 |
| median_mae | event | -0.04842502253247816 | -0.07945980991472545 | -0.02338892095927834 | False | 12 | 102 | 102 | 122400 | 10000 | 42 |
| pinball | non_event | -0.003226272101572789 | -0.01058049281249829 | 0.0030910686469516743 | True | 12 | 102 | 261 | 313200 | 10000 | 42 |
| median_mae | non_event | -0.007978437490171275 | -0.02705078249554132 | 0.008539570486736705 | True | 12 | 102 | 261 | 313200 | 10000 | 42 |

## Month-stratified difference-in-differences

This is `(XGBoost minus daily embedded transformer on event days) - (the same difference on month-matched non-event days)`. The bootstrap resamples complete days separately within each calendar-month/slice stratum. For relative pinball, each sampled model difference is divided by its sampled mean actual demand before the contrast is formed.

| metric | contrast | difference_in_differences | ci95_low | ci95_high | ci_excludes_zero | event_months | event_days_weight | bootstrap_draws | seed |
|---|---|---|---|---|---|---|---|---|---|
| pinball | (xgboost_minus_daily_embed)_event_minus_month_matched_non_event | -0.016369365170412806 | -0.029982491096280532 | -0.0042405449803329205 | True | 12 | 102 | 10000 | 42 |
| relative_pinball | (xgboost_minus_daily_embed)_event_minus_month_matched_non_event | -0.004267431277700022 | -0.00827968805889284 | -0.0005710198533704188 | True | 12 | 102 | 10000 | 42 |

## Reproduction

Command: `python -m src.event_day_analysis --draws 10000 --seed 42`

Wall-clock time: 1.03 minutes.
