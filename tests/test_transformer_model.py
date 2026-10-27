import numpy as np

from src.train_transformer import WindowDataset


def test_window_dataset_omits_static_feature_without_embedding():
    dataset = WindowDataset(np.zeros((1, 400), dtype="float32"), np.zeros((400, 2), dtype="float32"),
                            [(0, 336)], use_station_embedding=False)
    assert "static_categorical_features" not in dataset[0]
