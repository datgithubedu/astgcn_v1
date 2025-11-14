from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class CanonicalTrafficData:
    edge_df: pd.DataFrame
    weather_df: pd.DataFrame
    topology_df: pd.DataFrame

    speed_scaler: StandardScaler
    weather_scaler: StandardScaler

    train_run_ids: list
    val_run_ids: list
    test_run_ids: list

    @classmethod
    def from_parquet(cls, path: str, train_ratio=0.7, val_ratio=0.15):
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        run_ids = sorted(df["run_id"].unique())

        n_train = int(len(run_ids) * train_ratio)
        n_val = int(len(run_ids) * val_ratio)

        train_ids = run_ids[:n_train]
        val_ids = run_ids[n_train:n_train + n_val]
        test_ids = run_ids[n_train + n_val:]

        train_df = df[df["run_id"].isin(train_ids)]

        speed_scaler = StandardScaler().fit(train_df[["speed_kmh"]])
        weather_scaler = StandardScaler().fit(
            train_df[["temperature_c", "wind_speed_kmh", "precipitation_mm"]]
        )

        # Weather (unique per timestamp)
        weather_df = (
            df[["timestamp", "temperature_c", "wind_speed_kmh", "precipitation_mm"]]
            .drop_duplicates()
            .sort_values("timestamp")
        )

        topology_df = df[
            ["edge_id", "node_a_id", "node_b_id", "distance_m", "road_type"]
        ].drop_duplicates()

        return cls(
            edge_df=df,
            weather_df=weather_df,
            topology_df=topology_df,
            speed_scaler=speed_scaler,
            weather_scaler=weather_scaler,
            train_run_ids=train_ids,
            val_run_ids=val_ids,
            test_run_ids=test_ids,
        )
