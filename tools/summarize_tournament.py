from __future__ import annotations

from pathlib import Path
import re
import math

import pandas as pd
import torch


def _safe_max(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(max(values))


def _safe_last(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(values[-1])


def main() -> None:
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    checkpoints = sorted(output_dir.glob("*.pth"))

    pattern = re.compile(r"^(?P<model>.+)_(?P<loss>Baseline_CE|Hybrid_MSE)\.pth$")

    rows: list[dict[str, object]] = []
    for checkpoint_path in checkpoints:
        match = pattern.match(checkpoint_path.name)
        if not match:
            continue

        model = match.group("model")
        loss_key = match.group("loss")
        run = "Baseline (CE)" if loss_key == "Baseline_CE" else "Hybrid (50/50 CE+MSE)"

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        history = ckpt.get("history", {})

        train_exact = list(history.get("train_exact_acc", []))
        val_exact = list(history.get("val_exact_acc", []))
        train_tol = list(history.get("train_tol_acc", []))
        val_tol = list(history.get("val_tol_acc", []))
        train_loss = list(history.get("train_loss", []))
        val_loss = list(history.get("val_loss", []))

        row = {
            "Model": model,
            "Run": run,
            "Best Val Exact (%)": _safe_max(val_exact) * 100.0,
            "Best Val Tol (±1) (%)": _safe_max(val_tol) * 100.0,
            "Final Train Exact (%)": _safe_last(train_exact) * 100.0,
            "Final Val Exact (%)": _safe_last(val_exact) * 100.0,
            "Overfit Gap (Train-Val, pp)": (_safe_last(train_exact) - _safe_last(val_exact)) * 100.0,
            "Final Train Loss": _safe_last(train_loss),
            "Final Val Loss": _safe_last(val_loss),
            "Epochs": len(train_loss),
            "Checkpoint": checkpoint_path.as_posix(),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    model_order = ["MobileNet", "GoogLeNet", "InceptionV3", "VGG16", "VGG19"]
    run_order = ["Baseline (CE)", "Hybrid (50/50 CE+MSE)"]
    if not df.empty:
        df["Model"] = pd.Categorical(df["Model"], model_order, ordered=True)
        df["Run"] = pd.Categorical(df["Run"], run_order, ordered=True)
        df = df.sort_values(["Model", "Run"]).reset_index(drop=True)

    out_csv = output_dir / "tournament_summary.csv"
    df.to_csv(out_csv, index=False)

    # Print a compact table for quick copy/paste into docs.
    display_cols = [
        "Model",
        "Run",
        "Best Val Exact (%)",
        "Best Val Tol (±1) (%)",
        "Overfit Gap (Train-Val, pp)",
        "Final Train Loss",
        "Epochs",
    ]

    def fmt(x: object) -> str:
        if isinstance(x, (int,)):
            return str(x)
        if isinstance(x, (float,)):
            if math.isnan(x):
                return "nan"
            return f"{x:.2f}"
        return str(x)

    if df.empty:
        print("No checkpoints found in outputs/.")
        return

    printable = df[display_cols].copy()
    for col in printable.columns:
        printable[col] = printable[col].map(fmt)

    print(printable.to_string(index=False))
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
