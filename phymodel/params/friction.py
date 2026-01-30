from __future__ import annotations

from typing import Any, Dict, Tuple


def friction_params_from_payload(payload: Dict[str, Any], model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (params_for_model, full_friction_section)."""
    tunable = payload.get("tunable") if isinstance(payload, dict) else None
    friction = tunable.get("friction") if isinstance(tunable, dict) else None
    if not isinstance(friction, dict):
        return {}, {}
    section = friction.get(model)
    return (section if isinstance(section, dict) else {}), friction

