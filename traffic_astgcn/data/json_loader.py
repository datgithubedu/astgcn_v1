import json
from pathlib import Path
import pandas as pd


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_canonical_from_runs(
    runs_root: str | Path,
    output_parquet: str | Path = "data/processed/canonical.parquet",
) -> pd.DataFrame:
    """
    Convert thư mục runs/*/*.json → canonical.parquet
    """
    runs_root = Path(runs_root)
    run_dirs = sorted([d for d in runs_root.iterdir() if d.is_dir()])

    all_rows = []

    for run_idx, run_dir in enumerate(run_dirs):
        edges = _load_json(run_dir / "edges.json")
        traffics = _load_json(run_dir / "traffic_edges.json")
        weathers = _load_json(run_dir / "weather_snapshot.json")

        # Map edge (nodeA, nodeB)
        edge_lookup = {}
        for e in edges:
            key = (e["start_node_id"], e["end_node_id"])
            edge_lookup[key] = {
                "edge_id": e["edge_id"],
                "distance_m": e.get("distance_m", None),
                "road_type": e.get("road_type", None),
            }

        # Weather mean
        weather_df = pd.DataFrame(weathers)
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])
        mean_weather = {
            "temperature_c": weather_df["temperature_c"].mean(),
            "precipitation_mm": weather_df["precipitation_mm"].mean(),
            "wind_speed_kmh": weather_df["wind_speed_kmh"].mean(),
        }

        # Traffic samples
        for rec in traffics:
            node_a = rec["node_a_id"]
            node_b = rec["node_b_id"]
            key = (node_a, node_b)

            edge_info = edge_lookup.get(key, None)
            if edge_info is None:
                edge_id = f"{node_a}-{node_b}"
                distance_m = rec.get("distance_km", 0.0) * 1000.0
                road_type = None
            else:
                edge_id = edge_info["edge_id"]
                distance_m = edge_info["distance_m"]
                road_type = edge_info["road_type"]

            row = {
                "run_id": run_idx,
                "timestamp": pd.to_datetime(rec["timestamp"]),
                "edge_id": edge_id,
                "node_a_id": node_a,
                "node_b_id": node_b,
                "distance_m": distance_m,
                "road_type": road_type,
                "speed_kmh": rec["speed_kmh"],
                "temperature_c": mean_weather["temperature_c"],
                "precipitation_mm": mean_weather["precipitation_mm"],
                "wind_speed_kmh": mean_weather["wind_speed_kmh"],
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows).sort_values("timestamp").reset_index(drop=True)

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)

    print(f"Saved canonical: {output_parquet}")
    return df


if __name__ == "__main__":
    build_canonical_from_runs("data/raw/runs")
