# Calibration diagnostics

This is a validation-only diagnostic, not a test result. Reliability uses all saved 2025 validation rows. Post-hoc calibration fits one non-negative additive conformal central-interval adjustment on local 2025-01-01 through 2025-06-30 only, then scores the unchanged and adjusted forecasts on local 2025-07-01 through 2025-12-31 only; no row is used to both fit and score the adjustment. The adjustment widens quantiles around q50. It does not clip or sort predictions, so the calibrated forecast differs only by that one fitted adjustment.

## Held-out second-half scores

| model | variant | conformal_adjustment | pinball_loss | crps | coverage_90 | median_mae | n_observations | days |
|---|---|---|---|---|---|---|---|---|
| xgboost_shared_quantile | uncalibrated_second_half | 5.558772954827873e-06 | 0.5797105907818179 | 1.0806613784416665 | 0.9020628415300547 | 1.5569759197721469 | 219600 | 183 |
| xgboost_shared_quantile | conformal_second_half | 5.558772954827873e-06 | 0.5797106384706364 | 1.0806614680358184 | 0.9039617486338798 | 1.5569759197721469 | 219600 | 183 |
| hf_transformer_daily_embed_noweather | uncalibrated_second_half | 0.0 | 0.5840885048173712 | 1.0886727663934428 | 0.9411293260473589 | 1.566186247723133 | 219600 | 183 |
| hf_transformer_daily_embed_noweather | conformal_second_half | 0.0 | 0.5840885048173714 | 1.0886727663934426 | 0.9411293260473589 | 1.566186247723133 | 219600 | 183 |
| transformer_xgbfeatures_longcontext_nb | uncalibrated_second_half | 0.0 | 0.5892470999904132 | 1.098506466302368 | 0.9476593806921676 | 1.5802777777777777 | 219600 | 183 |
| transformer_xgbfeatures_longcontext_nb | conformal_second_half | 0.0 | 0.5892470999904131 | 1.098506466302368 | 0.9476593806921676 | 1.5802777777777777 | 219600 | 183 |
| transformer_xgbfeatures_longcontext_quantile | uncalibrated_second_half | 0.0 | 0.6036525485246136 | 1.125284731387574 | 0.9165437158469946 | 1.6206318564572837 | 219600 | 183 |
| transformer_xgbfeatures_longcontext_quantile | conformal_second_half | 0.0 | 0.6036525485246137 | 1.1252847313875742 | 0.9165437158469946 | 1.6206318564572837 | 219600 | 183 |

## Full-2025 reliability by quantile

| record_type | model | nominal_quantile | empirical_coverage |
|---|---|---|---|
| reliability_full_validation | xgboost_shared_quantile | 0.05 | 0.1918457300275482 |
| reliability_full_validation | xgboost_shared_quantile | 0.1 | 0.251421028466483 |
| reliability_full_validation | xgboost_shared_quantile | 0.15 | 0.2883815426997245 |
| reliability_full_validation | xgboost_shared_quantile | 0.2 | 0.3220087235996327 |
| reliability_full_validation | xgboost_shared_quantile | 0.25 | 0.35168044077134986 |
| reliability_full_validation | xgboost_shared_quantile | 0.3 | 0.38199265381083564 |
| reliability_full_validation | xgboost_shared_quantile | 0.35 | 0.4238085399449036 |
| reliability_full_validation | xgboost_shared_quantile | 0.4 | 0.4582254361799816 |
| reliability_full_validation | xgboost_shared_quantile | 0.45 | 0.4951056014692378 |
| reliability_full_validation | xgboost_shared_quantile | 0.5 | 0.533271349862259 |
| reliability_full_validation | xgboost_shared_quantile | 0.55 | 0.5740587695133149 |
| reliability_full_validation | xgboost_shared_quantile | 0.6 | 0.6165495867768596 |
| reliability_full_validation | xgboost_shared_quantile | 0.65 | 0.657584940312213 |
| reliability_full_validation | xgboost_shared_quantile | 0.7 | 0.6987695133149678 |
| reliability_full_validation | xgboost_shared_quantile | 0.75 | 0.744589072543618 |
| reliability_full_validation | xgboost_shared_quantile | 0.8 | 0.7923599632690542 |
| reliability_full_validation | xgboost_shared_quantile | 0.85 | 0.8394674012855831 |
| reliability_full_validation | xgboost_shared_quantile | 0.9 | 0.8871418732782369 |
| reliability_full_validation | xgboost_shared_quantile | 0.95 | 0.941354453627181 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.05 | 0.33093204775022955 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.1 | 0.3612465564738292 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.15 | 0.3929499540863177 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.2 | 0.4264302112029385 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.25 | 0.4598048668503214 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.3 | 0.4936225895316804 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.35 | 0.5270684113865932 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.4 | 0.5609664830119375 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.45 | 0.5955394857667585 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.5 | 0.6299678604224058 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.55 | 0.6650252525252526 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.6 | 0.7005463728191 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.65 | 0.7361455463728191 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.7 | 0.7725550964187328 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.75 | 0.8091460055096419 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.8 | 0.845257116620753 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.85 | 0.8815771349862259 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.9 | 0.9178443526170799 |
| reliability_full_validation | hf_transformer_daily_embed_noweather | 0.95 | 0.9554476584022038 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.05 | 0.3338108356290174 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.1 | 0.36529384756657485 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.15 | 0.39794306703397614 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.2 | 0.4308241505968779 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.25 | 0.4635904499540863 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.3 | 0.49659320477502294 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.35 | 0.5301101928374655 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.4 | 0.5634825528007347 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.45 | 0.5976469237832874 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.5 | 0.6320316804407714 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.55 | 0.6665312213039486 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.6 | 0.7017791551882461 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.65 | 0.7372222222222222 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.7 | 0.7730532598714417 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.75 | 0.8090197428833793 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.8 | 0.845325987144169 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.85 | 0.8826124885215795 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.9 | 0.920530303030303 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_nb | 0.95 | 0.9592493112947659 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.05 | 0.0315702479338843 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.1 | 0.08833792470156107 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.15 | 0.15639807162534436 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.2 | 0.22567033976124884 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.25 | 0.286935261707989 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.3 | 0.3468135904499541 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.35 | 0.39678145087235994 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.4 | 0.4422979797979798 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.45 | 0.48142332415059685 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.5 | 0.522757116620753 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.55 | 0.5618434343434343 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.6 | 0.597807621671258 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.65 | 0.6386042240587695 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.7 | 0.6802777777777778 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.75 | 0.7328282828282828 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.8 | 0.7701262626262626 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.85 | 0.8276423324150597 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.9 | 0.88732782369146 |
| reliability_full_validation | transformer_xgbfeatures_longcontext_quantile | 0.95 | 0.9437235996326906 |

Figure: `calibration_reliability_2025.png`.

## Reproduction

`python -m src.calibration_diagnostics`

Wall-clock time: 0.06 minutes.
