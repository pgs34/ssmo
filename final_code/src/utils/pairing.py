from __future__ import annotations

from typing import Any

METHOD_ALIASES = {
    "independent": "independent",
    "dml": "dml",
    "ssml": "ssml",
}

JOINT_METHODS = frozenset({"dml", "ssml"})


def canonicalize_method_name(method_name: str) -> str:
    key = str(method_name).lower()
    if key not in METHOD_ALIASES:
        raise ValueError(f"Unsupported method '{method_name}'. Use one of: {', '.join(sorted(METHOD_ALIASES))}")
    return METHOD_ALIASES[key]


def uses_peer_model(method_name: str) -> bool:
    return canonicalize_method_name(method_name) in JOINT_METHODS


def build_pair_metadata(
    model_name: str,
    peer_model_name: str | None,
) -> dict[str, Any]:
    if not peer_model_name:
        return {
            "pair_tag": model_name,
            "pair_type": "single",
            "peer_model": None,
            "is_joint_training": False,
            "is_heterogeneous_pair": False,
        }

    is_heterogeneous_pair = peer_model_name != model_name
    pair_type = "heterogeneous" if is_heterogeneous_pair else "homogeneous"
    pair_tag = f"{model_name}__{peer_model_name}" if is_heterogeneous_pair else model_name
    return {
        "pair_tag": pair_tag,
        "pair_type": pair_type,
        "peer_model": peer_model_name,
        "is_joint_training": True,
        "is_heterogeneous_pair": is_heterogeneous_pair,
    }
