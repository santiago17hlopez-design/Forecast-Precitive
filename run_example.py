"""End-to-end example run.

    python generate_sample_data.py
    python run_example.py

Expect the data alert to fire: the sample history contains a deliberately
damaged year, so the engine should detect it, trim the training window, and say
so loudly. That is the demonstration, not a failure.
"""
from __future__ import annotations

from forecast.pipeline import run_forecast

if __name__ == "__main__":
    summary = run_forecast(
        input_path="data/history.csv",
        output_path="output/forecast_results.csv",
        entity_code="DEMO",
        verbosity="full",
        overrides={
            # Two figures taken from a published report would go here, to turn
            # on the validation traffic light:
            # "CONTROL_POINTS": {"2024-09": 1234567.0},
        },
    )

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    for key, value in summary.items():
        if key in ("data_warning", "trimmed", "coverage"):
            continue
        print(f"  {key:22}: {value}")

    if summary.get("data_warning"):
        print(f"\n  data_warning          : "
              f"{len(summary['data_warning']['collapsed_months'])} months flagged, "
              f"years {summary['data_warning']['incomplete_years']}")
    if summary.get("trimmed"):
        print(f"  trimmed               : training from {summary['trimmed']['from']} "
              f"({summary['trimmed']['training_months']} months)")
