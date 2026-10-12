from penguin_species_classifier.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, PROCESSED_DATA_DIR

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_COLUMNS,
        ),
    ])