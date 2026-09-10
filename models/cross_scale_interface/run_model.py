import json
from pathlib import Path

from cross_scale_interface import run_analysis


def main():
    payload = {
        "symmetric": run_analysis(0.0),
        "heterogeneous": run_analysis(0.25),
    }
    target = Path(__file__).with_name("results.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
