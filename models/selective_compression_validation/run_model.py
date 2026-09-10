import json
from pathlib import Path

from selective_compression_validation import run_analysis


target = Path(__file__).with_name("results.json")
target.write_text(json.dumps(run_analysis(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(target)

