import os
from pathlib import Path
import sys

try:
    from dotenv import load_dotenv
    _dotenv_loaded = load_dotenv()
except Exception:
    _dotenv_loaded = False


# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

# Ensure src is on path for 'src' layout
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# Data and results locations (override via environment variables)
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
ANNOTATIONS_GLOB = os.getenv(
    "ANNOTATIONS_GLOB",
    str(DATA_ROOT / "annotations" / "*.jsonl"),
)

RESULTS_ROOT = Path(os.getenv("RESULTS_ROOT", PROJECT_ROOT / "results"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", RESULTS_ROOT / "Feb-2025 Results"))

# Optional: Map of display model names to CSV paths for results aggregation
# Override by setting MODEL_RESULTS_MAP as a JSON string in env if needed
DEFAULT_MODEL_RESULTS_MAP = {
    "Azure Health DeID v1": str(RESULTS_ROOT / "2024-04-11 Microsoft HDS output.csv"),
    "anoncat": str(RESULTS_ROOT / "2024-06-10 anoncat.csv"),
    "11 Feb 2025 OUH Fine Tuned AnonCAT 0.00002 10ep new concepts 10per": str(
        RESULTS_ROOT / "2025-02-11 10per FT-Anoncat New Concepts Added 0.00002 10ep.csv"
    ),
    "obi/deid_roberta_i2b2": str(RESULTS_ROOT / "2025-02-06 New OBI BERT benchmarks.csv"),
    "obi/deid_bert_i2b2": str(RESULTS_ROOT / "2025-02-06 New OBI BERT benchmarks.csv"),
}


def get_model_results_map() -> dict:
    import json

    env_val = os.getenv("MODEL_RESULTS_MAP")
    if env_val:
        try:
            parsed = json.loads(env_val)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return DEFAULT_MODEL_RESULTS_MAP


# API keys and endpoints (loaded from environment)
AZURE_HDS_BEARER_TOKEN = os.getenv("AZURE_HDS_BEARER_TOKEN")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "GPT35-turbo-base")

HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")


# Behavior flags
FAST_RESULTS = os.getenv("FAST_RESULTS", "false").lower() in {"1", "true", "yes"}


