import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib

# Ordinal Mappings
QUALITY_MAP = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0, np.nan: 0}
EXPOSURE_MAP = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0, np.nan: 0}
BSMT_FIN_MAP = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0, np.nan: 0}
GARAGE_FIN_MAP = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0, np.nan: 0}
FUNCTIONAL_MAP = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0, np.nan: 7}

NONE_CAT_COLS = [
    'Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'PoolQC', 'Fence', 'MiscFeature', 'MasVnrType'
]

ORDINAL_COLS = {
    'ExterQual': QUALITY_MAP,
    'ExterCond': QUALITY_MAP,
    'BsmtQual': QUALITY_MAP,
    'BsmtCond': QUALITY_MAP,
    'HeatingQC': QUALITY_MAP,
    'KitchenQual': QUALITY_MAP,
    'FireplaceQu': QUALITY_MAP,
    'GarageQual': QUALITY_MAP,
    'GarageCond': QUALITY_MAP,
    'PoolQC': QUALITY_MAP,
    'BsmtExposure': EXPOSURE_MAP,
    'BsmtFinType1': BSMT_FIN_MAP,
    'BsmtFinType2': BSMT_FIN_MAP,
    'GarageFinish': GARAGE_FIN_MAP,
    'Functional': FUNCTIONAL_MAP
}

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Robust, production-grade preprocessing pipeline tailored for housing data.
    Ensures consistent feature dimensions across folds and test sets.
    """
    def __init__(self, categories_dict: dict = None):
        self.neighborhood_lotfrontage_medians_ = {}
        self.num_medians_ = {}
        self.cat_modes_ = {}
        self.skewed_cols_ = []
        self.nominal_cols_ = []
        self.num_cols_ = []
        self.encoder_ = None
        self.scaler_ = None
        self.categories_dict_ = categories_dict
        self.feature_names_out_ = None
        self.is_fitted_ = False

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Handle None categoricals
        for col in NONE_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna('None')

        # Numerical defaults for missing features
        if 'MasVnrArea' in df.columns:
            df['MasVnrArea'] = df['MasVnrArea'].fillna(0.0)
        
        for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        if 'GarageYrBlt' in df.columns and 'YearBuilt' in df.columns:
            df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])

        # Ordinal encoding
        for col, mapping in ORDINAL_COLS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(float)

        # Engineered features
        bsmt_sf = df['TotalBsmtSF'] if 'TotalBsmtSF' in df.columns else 0
        flr1_sf = df['1stFlrSF'] if '1stFlrSF' in df.columns else 0
        flr2_sf = df['2ndFlrSF'] if '2ndFlrSF' in df.columns else 0
        df['TotalSF'] = bsmt_sf + flr1_sf + flr2_sf

        full_bath = df['FullBath'] if 'FullBath' in df.columns else 0
        half_bath = df['HalfBath'] if 'HalfBath' in df.columns else 0
        bsmt_full = df['BsmtFullBath'] if 'BsmtFullBath' in df.columns else 0
        bsmt_half = df['BsmtHalfBath'] if 'BsmtHalfBath' in df.columns else 0
        df['TotalBath'] = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

        yr_sold = df['YrSold'] if 'YrSold' in df.columns else 2010
        yr_built = df['YearBuilt'] if 'YearBuilt' in df.columns else 1970
        yr_remod = df['YearRemodAdd'] if 'YearRemodAdd' in df.columns else yr_built
        gar_built = df['GarageYrBlt'] if 'GarageYrBlt' in df.columns else yr_built

        df['HouseAge'] = (yr_sold - yr_built).clip(lower=0)
        df['RemodAge'] = (yr_sold - yr_remod).clip(lower=0)
        df['IsRemodeled'] = (yr_remod != yr_built).astype(float)
        df['GarageAge'] = (yr_sold - gar_built).clip(lower=0)

        wood_deck = df['WoodDeckSF'] if 'WoodDeckSF' in df.columns else 0
        open_porch = df['OpenPorchSF'] if 'OpenPorchSF' in df.columns else 0
        enc_porch = df['EnclosedPorch'] if 'EnclosedPorch' in df.columns else 0
        ssn_porch = df['3SsnPorch'] if '3SsnPorch' in df.columns else 0
        screen_porch = df['ScreenPorch'] if 'ScreenPorch' in df.columns else 0
        df['TotalPorchSF'] = wood_deck + open_porch + enc_porch + ssn_porch + screen_porch

        df['HasPool'] = (df['PoolArea'] > 0).astype(float) if 'PoolArea' in df.columns else 0.0
        df['HasGarage'] = (df['GarageArea'] > 0).astype(float) if 'GarageArea' in df.columns else 0.0
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(float) if 'TotalBsmtSF' in df.columns else 0.0
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(float) if 'Fireplaces' in df.columns else 0.0

        if 'OverallQual' in df.columns:
            df['Qual_x_TotalSF'] = df['OverallQual'] * df['TotalSF']
            df['Qual_x_GrLivArea'] = df['OverallQual'] * (df['GrLivArea'] if 'GrLivArea' in df.columns else 0)

        return df

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        if 'Id' in X.columns:
            X = X.drop(columns=['Id'])
        if 'SalePrice' in X.columns:
            X = X.drop(columns=['SalePrice'])

        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            self.neighborhood_lotfrontage_medians_ = X.groupby('Neighborhood')['LotFrontage'].median().to_dict()
            global_lf_median = X['LotFrontage'].median()
        else:
            global_lf_median = 68.0

        X_eng = self._engineer_features(X)

        if 'LotFrontage' in X_eng.columns and 'Neighborhood' in X_eng.columns:
            X_eng['LotFrontage'] = X_eng.apply(
                lambda row: self.neighborhood_lotfrontage_medians_.get(row['Neighborhood'], global_lf_median)
                if pd.isna(row['LotFrontage']) else row['LotFrontage'],
                axis=1
            )

        self.num_cols_ = X_eng.select_dtypes(include=[np.number]).columns.tolist()
        self.nominal_cols_ = X_eng.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.num_cols_:
            self.num_medians_[col] = X_eng[col].median()
            X_eng[col] = X_eng[col].fillna(self.num_medians_[col])

        for col in self.nominal_cols_:
            self.cat_modes_[col] = X_eng[col].mode()[0] if not X_eng[col].mode().empty else 'Missing'
            X_eng[col] = X_eng[col].fillna(self.cat_modes_[col])

        skewness = X_eng[self.num_cols_].skew()
        self.skewed_cols_ = skewness[abs(skewness) > 0.75].index.tolist()

        for col in self.skewed_cols_:
            X_eng[col] = np.log1p(np.maximum(0, X_eng[col]))

        if self.nominal_cols_:
            if self.categories_dict_ is not None:
                cats = [self.categories_dict_[col] for col in self.nominal_cols_]
                self.encoder_ = OneHotEncoder(categories=cats, sparse_output=False, handle_unknown='ignore')
            else:
                self.encoder_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            self.encoder_.fit(X_eng[self.nominal_cols_])
            encoded_names = self.encoder_.get_feature_names_out(self.nominal_cols_).tolist()
        else:
            encoded_names = []

        self.scaler_ = RobustScaler()
        self.scaler_.fit(X_eng[self.num_cols_])

        self.feature_names_out_ = self.num_cols_ + encoded_names
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("TabularPreprocessor must be fitted before calling transform.")

        X = X.copy()
        if 'Id' in X.columns:
            X = X.drop(columns=['Id'])
        if 'SalePrice' in X.columns:
            X = X.drop(columns=['SalePrice'])

        X_eng = self._engineer_features(X)

        global_lf_median = np.median(list(self.neighborhood_lotfrontage_medians_.values())) if self.neighborhood_lotfrontage_medians_ else 68.0
        if 'LotFrontage' in X_eng.columns and 'Neighborhood' in X_eng.columns:
            X_eng['LotFrontage'] = X_eng.apply(
                lambda row: self.neighborhood_lotfrontage_medians_.get(row['Neighborhood'], global_lf_median)
                if pd.isna(row['LotFrontage']) else row['LotFrontage'],
                axis=1
            )

        for col in self.num_cols_:
            if col in X_eng.columns:
                X_eng[col] = X_eng[col].fillna(self.num_medians_.get(col, 0.0))
            else:
                X_eng[col] = self.num_medians_.get(col, 0.0)

        for col in self.nominal_cols_:
            if col in X_eng.columns:
                X_eng[col] = X_eng[col].fillna(self.cat_modes_.get(col, 'Missing'))
            else:
                X_eng[col] = self.cat_modes_.get(col, 'Missing')

        for col in self.skewed_cols_:
            X_eng[col] = np.log1p(np.maximum(0, X_eng[col]))

        num_scaled = self.scaler_.transform(X_eng[self.num_cols_])

        if self.nominal_cols_ and self.encoder_ is not None:
            cat_encoded = self.encoder_.transform(X_eng[self.nominal_cols_])
            X_out = np.hstack([num_scaled, cat_encoded])
        else:
            X_out = num_scaled

        return X_out.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'TabularPreprocessor':
        return joblib.load(filepath)
