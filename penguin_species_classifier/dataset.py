
import pandas as pd
from penguin_species_classifier.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)


def clean_data(data):
    return data.dropna()
   
def split_features_target(data):
    X = data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return X, y


def split_data(X, y, seed, test_size=0.2):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

