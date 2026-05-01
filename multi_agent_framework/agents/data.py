"""
DataAgent — 数据清洗、归一化、特征工程

核心能力:
  - 支持 CSV/Excel/Parquet 多格式加载
  - 缺失值检测与处理 (均值/中位数/KNN 填充)
  - 异常值检测 (IQR / Z-score / Isolation Forest)
  - Z-score 归一化、Min-Max 缩放、Log 变换、Box-Cox
  - 特征降噪 (PCA 去噪、低方差过滤)
  - 描述性统计与数据画像生成
"""

from __future__ import annotations

import json
from typing import Any

from ..core.agent import BaseAgent
from ..core.message import Task, AgentRole


DATA_SYSTEM_PROMPT = """你是一个生物数据科学专家，专精于高通量表型数据的预处理与特征工程。

你的职责:
1. 加载多格式数据文件 (CSV/Excel/Parquet)
2. 执行数据质量诊断（缺失值、异常值、分布偏度）
3. 标准化处理：Z-score 归一化、Log 变换、Box-Cox
4. 特征筛选：低方差过滤、相关性分析、PCA 降维降噪
5. 输出清洗后的数据摘要与处理报告

输出格式：返回处理步骤、统计摘要和清洗后的数据（限制行数）
"""


class DataAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(role=AgentRole.DATA, **kwargs)

    def _build_system_prompt(self) -> str:
        return DATA_SYSTEM_PROMPT

    async def _process_task(self, task: Task) -> str:
        title = task.title.lower()
        description = task.description

        if any(kw in title for kw in ["加载", "探索", "eda", "exploratory", "load"]):
            return self._data_exploration_template(task)
        elif any(kw in title for kw in ["清洗", "标准化", "归一", "normalize", "transform", "clean"]):
            return self._data_cleaning_template(task)
        elif any(kw in title for kw in ["特征", "降维", "降噪", "pca", "feature", "筛选"]):
            return self._feature_engineering_template(task)
        else:
            return self._data_exploration_template(task)

    def _data_exploration_template(self, task: Task) -> str:
        return '''```python
"""
数据加载与探索性分析 — 为生物表型组数据设计
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import json


class PhenoDataExplorer:
    """表型数据探索器"""

    def __init__(self, data_path: Optional[str] = None, df: Optional[pd.DataFrame] = None):
        if df is not None:
            self.df = df
        elif data_path:
            self.df = self._load_data(data_path)
        else:
            self.df = pd.DataFrame()
        self.report: dict = {}

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

    def summary(self) -> str:
        df = self.df
        self.report = {
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "dtypes": df.dtypes.value_counts().to_dict(),
            "missing": df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        numeric = df.select_dtypes(include=[np.number])
        if not numeric.empty:
            desc = numeric.describe()
            self.report["numeric_stats"] = {
                "count": int(df.shape[0]),
                "features_with_negatives": int((numeric.min() < 0).sum()),
                "features_with_zeros": int((numeric.min() == 0).sum()),
                "features_needing_log": int((numeric.skew() > 2).sum()),
            }

        return json.dumps(self.report, indent=2, ensure_ascii=False)

    def display_head(self, n: int = 10) -> str:
        return self.df.head(n).to_string()

    def missing_report(self) -> str:
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        report = pd.DataFrame({
            "missing_count": missing,
            "missing_pct": missing_pct,
            "dtype": self.df.dtypes,
        })
        return report[report["missing_count"] > 0].to_string()
```'''

    def _data_cleaning_template(self, task: Task) -> str:
        return '''```python
"""
数据清洗与标准化处理流水线
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from typing import Optional, Literal


class PhenoDataCleaner:
    """表型数据清洗流水线"""

    def __init__(self, df: pd.DataFrame, id_cols: Optional[list[str]] = None):
        self.df = df.copy()
        self.id_cols = id_cols or []
        self.numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                             if c not in self.id_cols]
        self.cleaning_log: list[str] = []

    def handle_missing(
        self,
        strategy: Literal["mean", "median", "drop", "knn"] = "median",
        threshold: float = 0.5,
    ) -> "PhenoDataCleaner":
        # 删除缺失率过高的列
        missing_pct = self.df[self.numeric_cols].isnull().mean()
        drop_cols = missing_pct[missing_pct > threshold].index.tolist()
        if drop_cols:
            self.df.drop(columns=drop_cols, inplace=True)
            self.numeric_cols = [c for c in self.numeric_cols if c not in drop_cols]
            self.cleaning_log.append(f"删除 {len(drop_cols)} 个缺失率 > {threshold} 的列")

        if strategy == "drop":
            before = len(self.df)
            self.df.dropna(subset=self.numeric_cols, inplace=True)
            self.cleaning_log.append(f"删除 {before - len(self.df)} 个含缺失值的行")
        elif strategy in ("mean", "median"):
            fill_val = self.df[self.numeric_cols].median() if strategy == "median" \
                       else self.df[self.numeric_cols].mean()
            self.df[self.numeric_cols] = self.df[self.numeric_cols].fillna(fill_val)
            self.cleaning_log.append(f"使用 {strategy} 填充缺失值")
        elif strategy == "knn":
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=5)
            self.df[self.numeric_cols] = imputer.fit_transform(self.df[self.numeric_cols])
            self.cleaning_log.append("使用 KNN (k=5) 填充缺失值")
        return self

    def remove_outliers(
        self,
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float = 3.0,
    ) -> "PhenoDataCleaner":
        if method == "zscore":
            z = np.abs(stats.zscore(self.df[self.numeric_cols], nan_policy="omit"))
            mask = (z < threshold).all(axis=1)
        else:
            Q1 = self.df[self.numeric_cols].quantile(0.25)
            Q3 = self.df[self.numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.df[self.numeric_cols] < (Q1 - 1.5 * IQR)) |
                     (self.df[self.numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

        removed = (~mask).sum()
        if removed > 0:
            self.df = self.df[mask].reset_index(drop=True)
            self.cleaning_log.append(f"使用 {method} 移除 {removed} 个异常值")
        return self

    def normalize(
        self,
        method: Literal["zscore", "minmax", "robust"] = "zscore",
    ) -> "PhenoDataCleaner":
        if method == "zscore":
            scaler = StandardScaler()
            self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])
        elif method == "minmax":
            scaler = MinMaxScaler()
            self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])
        self.cleaning_log.append(f"执行 {method} 归一化")
        return self

    def log_transform(self, offset: float = 1.0) -> "PhenoDataCleaner":
        # 仅对偏度 > 1 且非负的列执行 log 变换
        for col in self.numeric_cols:
            if self.df[col].min() >= 0 and abs(self.df[col].skew()) > 1:
                self.df[col] = np.log1p(self.df[col] * offset)
        self.cleaning_log.append("执行 log(1+x) 变换（偏度 > 1 的特征）")
        return self

    def boxcox_transform(self) -> "PhenoDataCleaner":
        try:
            pt = PowerTransformer(method="box-cox", standardize=False)
            # 仅对正值特征
            positive_cols = [c for c in self.numeric_cols if (self.df[c] > 0).all()]
            if positive_cols:
                self.df[positive_cols] = pt.fit_transform(self.df[positive_cols])
                self.cleaning_log.append(f"对 {len(positive_cols)} 个正值特征执行 Box-Cox")
        except Exception as e:
            self.cleaning_log.append(f"Box-Cox 失败: {e}")
        return self

    def pipeline(self) -> pd.DataFrame:
        return (
            self
            .handle_missing(strategy="median", threshold=0.5)
            .remove_outliers(method="iqr")
            .normalize(method="zscore")
            .log_transform()
            .df
        )

    def get_report(self) -> str:
        return "\n".join(self.cleaning_log)
```'''

    def _feature_engineering_template(self, task: Task) -> str:
        return '''```python
"""
特征工程 — 降噪、降维、特征筛选
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import Optional


class PhenoFeatureEngineer:
    """表型特征工程流水线"""

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.X = df.drop(columns=[target_col]) if target_col else df
        self.y = df[target_col] if target_col else None
        self.numeric_cols = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.X_num = self.X[self.numeric_cols].copy()
        self.log: list[str] = []

    def filter_low_variance(self, threshold: float = 0.01) -> "PhenoFeatureEngineer":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_num)
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X_scaled)
        kept_idx = selector.get_support(indices=True)
        removed = [c for i, c in enumerate(self.numeric_cols) if i not in kept_idx]
        self.X_num = self.X_num.iloc[:, kept_idx]
        self.numeric_cols = self.X_num.columns.tolist()
        if removed:
            self.log.append(f"移除 {len(removed)} 个低方差特征 (< {threshold})")
        return self

    def filter_high_correlation(self, threshold: float = 0.95) -> "PhenoFeatureEngineer":
        corr_matrix = self.X_num.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > threshold)]
        if drop_cols:
            self.X_num.drop(columns=drop_cols, inplace=True)
            self.numeric_cols = self.X_num.columns.tolist()
            self.log.append(f"移除 {len(drop_cols)} 个高相关特征 (r > {threshold})")
        return self

    def mutual_info_select(self, top_k: int = 100) -> "PhenoFeatureEngineer":
        if self.y is None:
            self.log.append("无目标列，跳过互信息筛选")
            return self

        mi_scores = mutual_info_regression(self.X_num.fillna(0), self.y, random_state=42)
        top_indices = np.argsort(mi_scores)[-top_k:]
        self.X_num = self.X_num.iloc[:, top_indices]
        self.numeric_cols = self.X_num.columns.tolist()
        self.log.append(f"基于互信息保留 Top {top_k} 个特征")
        return self

    def pca_denoise(self, n_components: float = 0.95) -> np.ndarray:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X_num.fillna(0))
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        self.log.append(
            f"PCA 降噪: {self.X_num.shape[1]} → {X_pca.shape[1]} 维 "
            f"(保留 {n_components*100:.0f}% 方差)"
        )
        return X_pca

    def get_report(self) -> str:
        return "\n".join(self.log) if self.log else "无特征工程操作"
```'''
