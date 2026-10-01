"""全局配置, 支持 ENV_MVFORGE_* 环境变量覆盖."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    """MVForge 运行配置.

    所有数值型参数均可经环境变量 ``ENV_MVFORGE_<NAME>`` 覆盖,
    例如 ``ENV_MVFORGE_RANDOM_STATE=7`` 会改写 random_state.
    """

    random_state: int = 42
    cca_reg: float = 1e-4
    cca_components: int = 10
    kcca_gamma: float = 1.0
    kcca_reg: float = 1e-3
    fusion_n_estimators: int = 200
    retrieval_topk: int = 5
    non_inferiority_tol: float = 0.02
    verbose: bool = False

    _ENV_PREFIX = "ENV_MVFORGE_"
    _MAP = {
        "RANDOM_STATE": "random_state",
        "CCA_REG": "cca_reg",
        "CCA_COMPONENTS": "cca_components",
        "KCCA_GAMMA": "kcca_gamma",
        "KCCA_REG": "kcca_reg",
        "FUSION_N_ESTIMATORS": "fusion_n_estimators",
        "RETRIEVAL_TOPK": "retrieval_topk",
        "NON_INFERIORITY_TOL": "non_inferiority_tol",
        "VERBOSE": "verbose",
    }

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for env_key, attr in cls._MAP.items():
            raw = os.environ.get(cls._ENV_PREFIX + env_key)
            if raw is None:
                continue
            cur = getattr(cfg, attr)
            try:
                if isinstance(cur, bool):
                    setattr(cfg, attr, raw.lower() in ("1", "true", "yes", "on"))
                elif isinstance(cur, int):
                    setattr(cfg, attr, int(raw))
                else:
                    setattr(cfg, attr, float(raw))
            except ValueError as exc:  # pragma: no cover
                from .errors import ConfigError

                raise ConfigError(
                    f"无法解析环境变量 {cls._ENV_PREFIX}{env_key}={raw!r}"
                ) from exc
        return cfg
