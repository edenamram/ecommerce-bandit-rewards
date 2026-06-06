"""
preprocessing.py
----------------
Loads and prepares the RetailRocket e-commerce dataset for offline
contextual bandit evaluation.

Dataset structure:
  events.csv  — user interaction logs (view, addtocart, transaction)
  item_properties_part1/2.csv — item feature snapshots over time

Output:
  A DataFrame where each row is one bandit "trial":
    - context vector  : user + item features (for LinUCB)
    - displayed_item  : the item that was actually shown
    - event_type      : view / addtocart / transaction
    - reward_binary   : R1
    - reward_weighted : R2
    - reward_maxstage : R3
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Funnel stage encoding
# ---------------------------------------------------------------------------
FUNNEL_ORDER = {"view": 0, "addtocart": 1, "transaction": 2}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_prepare(
    data_dir: str = "data/raw",
    n_items: int = 50,          # keep top-N most interacted items as "arms"
    n_components: int = 10,     # PCA dims for feature vectors
    min_events: int = 5,        # drop items with fewer events
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Main entry point.  Returns a cleaned trial DataFrame ready for simulation.

    Parameters
    ----------
    data_dir     : folder containing the raw CSV files from Kaggle
    n_items      : number of items to keep as bandit arms
    n_components : dimensionality of context vectors after PCA
    min_events   : minimum interaction count per item
    random_state : for reproducibility

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, visitorid, itemid,
        event_type, funnel_rank,
        reward_binary, reward_weighted, reward_maxstage,
        context  (numpy array of shape [n_components])
    """
    print("[preprocessing] Loading raw data …")
    events, props = _load_raw(data_dir)

    print("[preprocessing] Selecting top items …")
    top_items = _select_top_items(events, n_items, min_events)
    events = events[events["itemid"].isin(top_items)].copy()

    print(f"[preprocessing] Kept {len(top_items)} items, {len(events):,} events")

    print("[preprocessing] Building item feature matrix …")
    item_features = _build_item_features(props, top_items, n_components, random_state)

    print("[preprocessing] Building user feature vectors …")
    user_features = _build_user_features(events, n_components)

    print("[preprocessing] Merging & computing rewards …")
    trials = _build_trials(events, item_features, user_features)

    print(f"[preprocessing] Done. {len(trials):,} trials ready.\n")
    return trials


def get_arm_list(trials: pd.DataFrame) -> list:
    """Return sorted list of unique item IDs (arms)."""
    return sorted(trials["itemid"].unique().tolist())


def get_context_dim(trials: pd.DataFrame) -> int:
    """Return dimensionality of the context vectors."""
    return len(trials["context"].iloc[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw(data_dir: str):
    """Load events.csv and item_properties CSVs."""
    events_path = os.path.join(data_dir, "events.csv")
    if not os.path.exists(events_path):
        raise FileNotFoundError(
            f"Could not find {events_path}.\n"
            "Download the RetailRocket dataset from Kaggle and place the CSV "
            "files in data/raw/"
        )

    events = pd.read_csv(events_path)
    # Normalise column names (Kaggle export may vary)
    events.columns = [c.strip().lower() for c in events.columns]
    # Required columns: timestamp, visitorid, event, itemid
    events = events.rename(columns={"event": "event_type"})
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["timestamp", "visitorid", "itemid", "event_type"])
    events["itemid"] = events["itemid"].astype(int)
    events["visitorid"] = events["visitorid"].astype(int)
    events = events.sort_values("timestamp").reset_index(drop=True)

    # Item properties (optional — used for richer features)
    props_parts = []
    for part in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        p = os.path.join(data_dir, part)
        if os.path.exists(p):
            df = pd.read_csv(p)
            df.columns = [c.strip().lower() for c in df.columns]
            props_parts.append(df)
    props = pd.concat(props_parts, ignore_index=True) if props_parts else pd.DataFrame()

    return events, props


def _select_top_items(events: pd.DataFrame, n_items: int, min_events: int) -> list:
    """Keep the n_items most interacted items that pass min_events threshold."""
    counts = events.groupby("itemid").size()
    counts = counts[counts >= min_events]
    top = counts.nlargest(n_items).index.tolist()
    return top


def _build_item_features(
    props: pd.DataFrame,
    item_ids: list,
    n_components: int,
    random_state: int,
) -> dict:
    """
    Build a feature vector per item.

    If item properties are available we pivot the most common property values
    into a sparse matrix and reduce with PCA.
    Otherwise we fall back to a random (but fixed) embedding — this is a
    common trick in offline bandit evaluation when features are not available.
    """
    n_items = len(item_ids)

    if props.empty:
        # Fallback: deterministic random features seeded by item id
        rng = np.random.default_rng(random_state)
        features = {
            iid: rng.standard_normal(n_components)
            for iid in item_ids
        }
        return features

    # Filter to our selected items
    props = props[props["itemid"].isin(item_ids)].copy()
    props["itemid"] = props["itemid"].astype(int)

    # Use most frequent properties as binary features
    top_props = (
        props.groupby("property")
        .size()
        .nlargest(100)
        .index.tolist()
    )
    props = props[props["property"].isin(top_props)]

    # Pivot: rows=items, cols=properties (presence = 1)
    pivot = (
        props.groupby(["itemid", "property"])
        .size()
        .unstack(fill_value=0)
        .reindex(item_ids, fill_value=0)
    )

    X = pivot.values.astype(float)
    X = StandardScaler().fit_transform(X)

    # Reduce dimensionality
    actual_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=actual_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    # Pad with zeros if we got fewer components than requested
    if actual_components < n_components:
        pad = np.zeros((X_reduced.shape[0], n_components - actual_components))
        X_reduced = np.hstack([X_reduced, pad])

    features = {
        int(iid): X_reduced[i]
        for i, iid in enumerate(item_ids)
    }
    return features


def _build_user_features(events: pd.DataFrame, n_components: int) -> dict:
    """
    Build a feature vector per user from their historical event profile.

    Features:
      - fraction of events that are views / addtocarts / transactions
      - log1p(total events)
      - normalised recency of last event
    These are then padded/truncated to n_components.
    """
    agg = (
        events.groupby("visitorid")["event_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reindex(columns=["view", "addtocart", "transaction"], fill_value=0.0)
    )
    total = events.groupby("visitorid").size().rename("total")
    last_ts = events.groupby("visitorid")["timestamp"].max().rename("last_ts")

    profile = agg.join(total).join(last_ts)
    profile["log_total"] = np.log1p(profile["total"])
    ts_min = profile["last_ts"].min()
    ts_max = profile["last_ts"].max()
    profile["recency"] = (profile["last_ts"] - ts_min) / max(ts_max - ts_min, 1)
    profile = profile.drop(columns=["total", "last_ts"])

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(profile.values)

    # Pad or truncate to n_components
    if X.shape[1] < n_components:
        pad = np.zeros((X.shape[0], n_components - X.shape[1]))
        X = np.hstack([X, pad])
    else:
        X = X[:, :n_components]

    features = {
        int(uid): X[i]
        for i, uid in enumerate(profile.index)
    }
    return features


def _build_trials(
    events: pd.DataFrame,
    item_features: dict,
    user_features: dict,
) -> pd.DataFrame:
    """
    Combine events with features and compute all three reward signals.
    Each row = one bandit trial.
    """
    df = events.copy()

    # Funnel rank
    df["funnel_rank"] = df["event_type"].map(FUNNEL_ORDER).fillna(0).astype(int)

    # ---- R1: Binary reward ------------------------------------------------
    df["reward_binary"] = (df["event_type"] == "transaction").astype(float)

    # ---- R2: Weighted reward ----------------------------------------------
    WEIGHTS = {"view": 0.1, "addtocart": 0.5, "transaction": 1.0}
    df["reward_weighted"] = df["event_type"].map(WEIGHTS).fillna(0.0)

    # ---- R3: Max-stage reward per (visitor, session) ----------------------
    # For each visitor+item pair, find the deepest funnel stage reached,
    # then assign that stage's reward to every event in that pair.
    max_stage = (
        df.groupby(["visitorid", "itemid"])["funnel_rank"]
        .transform("max")
    )
    STAGE_REWARD = {0: 0.1, 1: 0.5, 2: 1.0}
    df["reward_maxstage"] = max_stage.map(STAGE_REWARD)

    # ---- Context vector = concat(user_feat, item_feat) --------------------
    n_comp = len(next(iter(item_features.values())))

    def make_context(row):
        u = user_features.get(row["visitorid"], np.zeros(n_comp))
        i = item_features.get(row["itemid"], np.zeros(n_comp))
        return np.concatenate([u, i])

    df["context"] = df.apply(make_context, axis=1)

    # Keep only rows where we have both user and item features
    valid_users = set(user_features.keys())
    valid_items = set(item_features.keys())
    df = df[
        df["visitorid"].isin(valid_users) & df["itemid"].isin(valid_items)
    ].copy()

    return df.reset_index(drop=True)
