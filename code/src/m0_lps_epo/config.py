from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union


@dataclass
class SimulationConfig:
    N: int
    T: int
    R: int
    W: int
    epsilon_type: str
    epsilon: Optional[float] = None
    epsilon_scale: Optional[float] = None


@dataclass
class NetworkConfig:
    type: str
    m_BA: int
    directed: bool = False
    p_ER: Optional[float] = None
    k_WS: Optional[int] = None
    p_WS: float = 0.1
    homophily_chi: float = 0.0
    attachment_epsilon: float = 1.0


@dataclass
class PrivateOpinionConfig:
    distribution: str
    mean: float
    sigma_x: float
    lower: float
    upper: float
    beta_a: Optional[float] = None
    beta_b: Optional[float] = None
    mixture_weight_left: float = 0.5
    left_mean: Optional[float] = None
    left_sigma: Optional[float] = None
    right_mean: Optional[float] = None
    right_sigma: Optional[float] = None
    empirical_path: Optional[str] = None
    empirical_value_col: str = "x"
    empirical_weight_col: str = "weight"


@dataclass
class CostConfig:
    distribution: str
    low: float = 0.0
    high: float = 1.0


@dataclass
class ProfileConfig:
    lambda_b: float
    sigma_b: float
    lambda_pi: float
    sigma_pi: float
    static: bool = True


@dataclass
class ActivityConfig:
    sigma_A: float
    rho_kq: float
    lambda_0: float
    omega: float
    posting_distribution: str = "poisson"
    nb_r: Optional[float] = None


@dataclass
class ParticipationConfig:
    theta_0: float
    theta_M: float
    theta_H: float
    theta_C: float
    eta: float
    theta_minor: float = 0.0
    minority_definition: str = "none"
    congruence_type: str = "item"


@dataclass
class ExpressionConfig:
    target: str
    alpha: float
    sigma_y: float
    tau_S: float = 0.1
    delta_S: float = 0.0
    n_dom_min: int = 3
    kappa: int = 2


@dataclass
class VisibilityConfig:
    K: int
    beta_E: float
    beta_P: float
    actual_visibility: str
    beta_G: float = 0.0
    popularity_signal: str = "none"


@dataclass
class MetricsConfig:
    delta_m: float
    delta_e: float
    delta_maj: float
    compute_fvb: bool
    fvb_pairs_L: int
    transition_bins_Q: int
    delta_B: Optional[float] = None
    bpw_p: float = 1.0


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str


@dataclass
class FullConfig:
    experiment: ExperimentConfig
    simulation: SimulationConfig
    network: NetworkConfig
    private_opinion: PrivateOpinionConfig
    cost: CostConfig
    profile: ProfileConfig
    activity: ActivityConfig
    participation: ParticipationConfig
    expression: ExpressionConfig
    visibility: VisibilityConfig
    metrics: MetricsConfig


def _parse_scalar(raw: str) -> Any:
    value = _strip_inline_comment(raw).strip()
    if value == "":
        return {}
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in {"null", "none"}:
        return None
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _strip_inline_comment(text: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return text[:idx]
    return text


def _load_simple_yaml(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load the small two-level YAML subset used by this project."""
    data: Dict[str, Dict[str, Any]] = {}
    current: Optional[str] = None

    # utf-8-sig accepts both ordinary UTF-8 and Windows-edited files with a BOM.
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" "):
            if not stripped.endswith(":"):
                raise ValueError(f"Expected section header at {path}:{line_no}")
            current = stripped[:-1]
            data[current] = {}
            continue
        if current is None:
            raise ValueError(f"Key outside section at {path}:{line_no}")
        if ":" not in stripped:
            raise ValueError(f"Expected key/value pair at {path}:{line_no}")
        key, value = stripped.split(":", 1)
        data[current][key.strip()] = _parse_scalar(value)

    return data


def load_config(path: Union[str, Path]) -> FullConfig:
    raw = _load_simple_yaml(Path(path))
    return FullConfig(
        experiment=ExperimentConfig(**raw["experiment"]),
        simulation=SimulationConfig(**raw["simulation"]),
        network=NetworkConfig(**raw["network"]),
        private_opinion=PrivateOpinionConfig(**raw["private_opinion"]),
        cost=CostConfig(**raw["cost"]),
        profile=ProfileConfig(**raw["profile"]),
        activity=ActivityConfig(**raw["activity"]),
        participation=ParticipationConfig(**raw["participation"]),
        expression=ExpressionConfig(**raw["expression"]),
        visibility=VisibilityConfig(**raw["visibility"]),
        metrics=MetricsConfig(**raw["metrics"]),
    )
