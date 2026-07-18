import json
import re
import os

from matplotlib.pylab import choice

out_path_train = "./dataset/train.jsonl"
out_path_test = "./dataset/test.jsonl"
out_path_validation = "./dataset/validation.jsonl"

# Postprocessing functions for text simplification

# only for tests
train_odds = 0.7
test_odds = 0.15
validation_odds = 0.15
for i in range(10):
    out_path_data = choice(
        [out_path_train, out_path_test, out_path_validation],
        p=[train_odds, test_odds, validation_odds]
    )

    print(f"Selected output path: {out_path_data}")
    