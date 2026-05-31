"""Map open-ended contextual driver text onto auditable macro channels."""

from __future__ import annotations

import json
import re


MACRO_CHANNEL_PATTERNS = {
    "macro_cpi_yoy": [
        r"\binflation\b",
        r"\bcpi\b",
        r"\bconsumer prices?\b",
        r"\bcost of living\b",
        r"\bprice pressures?\b",
    ],
    "macro_fed_funds_rate": [
        r"\bfed\b",
        r"\bfederal reserve\b",
        r"\binterest rates?\b",
        r"\bmonetary policy\b",
        r"\brate (?:cuts?|hikes?)\b",
        r"\btightening\b",
        r"\beasing\b",
    ],
    "macro_treasury_10y_2y_spread": [
        r"\byield curve\b",
        r"\btreasur(?:y|ies)\b",
        r"\btreasury spread\b",
        r"\bbond yields?\b",
    ],
    "macro_vix": [
        r"\bvix\b",
        r"\bvolatility\b",
        r"\brisk aversion\b",
        r"\bfear index\b",
    ],
    "macro_wti_oil": [
        r"\boil prices?\b",
        r"\bcrude(?: oil)?\b",
        r"\bwti\b",
        r"\benergy prices?\b",
    ],
    "macro_unemployment_rate": [
        r"\bunemployment\b",
        r"\bjob market\b",
        r"\blabou?r market\b",
        r"\bpayrolls?\b",
        r"\bemployment (?:data|report|figures|growth)\b",
        r"\bjobs report\b",
    ],
}

MACRO_CHANNEL_FEATURES = [
    f"driver_{variable}_channel_importance" for variable in MACRO_CHANNEL_PATTERNS
]


def _clamp01(value, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return min(1.0, max(0.0, numeric))


def _driver_text(driver: dict) -> str:
    fragments = [
        str(driver.get("driver_label", "") or ""),
        str(driver.get("driver_summary", "") or ""),
    ]
    for channel in driver.get("relevant_channels", []) or []:
        if isinstance(channel, dict):
            fragments.append(str(channel.get("channel", "") or ""))
            fragments.append(str(channel.get("available_variable", "") or ""))
    return " | ".join(fragments)


def map_macro_channels(drivers: list[dict]) -> tuple[dict[str, float], list[dict]]:
    importance_by_variable = {variable: 0.0 for variable in MACRO_CHANNEL_PATTERNS}
    matches = []
    for driver in drivers:
        if not isinstance(driver, dict):
            continue
        text = _driver_text(driver)
        channel_importance = max(
            [
                _clamp01(channel.get("importance"))
                for channel in driver.get("relevant_channels", []) or []
                if isinstance(channel, dict)
            ]
            or [0.0]
        )
        fallback_importance = max(
            _clamp01(driver.get("confidence")),
            _clamp01(driver.get("materiality")),
        )
        importance = max(channel_importance, fallback_importance)
        for variable, patterns in MACRO_CHANNEL_PATTERNS.items():
            matched_patterns = [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]
            if variable in text:
                matched_patterns.append(f"exact:{variable}")
            if not matched_patterns:
                continue
            importance_by_variable[variable] = max(importance_by_variable[variable], importance)
            matches.append(
                {
                    "macro_variable": variable,
                    "importance": importance,
                    "driver_label": str(driver.get("driver_label", "") or "")[:200],
                    "matched_patterns": matched_patterns,
                }
            )
    features = {
        f"driver_{variable}_channel_importance": float(importance)
        for variable, importance in importance_by_variable.items()
    }
    features["driver_macro_variable_channel_importance"] = float(
        max(importance_by_variable.values(), default=0.0)
    )
    return features, matches


def enrich_macro_channel_features(df):
    rows = []
    for value in df["dominant_drivers_json"].fillna("[]"):
        try:
            drivers = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            drivers = []
        features, matches = map_macro_channels(drivers if isinstance(drivers, list) else [])
        features["driver_macro_channel_matches_json"] = json.dumps(matches)
        rows.append(features)
    enriched = df.copy()
    features_df = type(df)(rows, index=df.index)
    for column in features_df.columns:
        enriched[column] = features_df[column]
    return enriched
