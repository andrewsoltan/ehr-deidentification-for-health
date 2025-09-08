## Benchmarking transformer-based models for medical record deidentification: A single centre, multi-specialty evaluation

**Repository**: andrewsoltan/ehr-deidentification-for-health
**Purpose**: This repository accompanies the academic manuscript, sharing code, configuration, and notebooks to run inference, evaluate models, and reproduce figures and metrics.

### Authors
Rachel Kuo MB BChir¹ ²*, Andrew A.S. Soltan MB BChir² ³ ⁴ ⁵*, Ciaran O’Hanlon MBBS¹ ², Alan Hasanic MBBS¹ ², David A. Clifton DPhil⁵ ⁶, Gary Collins Ph.D.¹ ⁷ ⁸, Dominic Furniss DM FRCS(Plast)¹ ²**, David W. Eyre DPhil BM BCh² ⁹ ¹⁰ ¹¹**

\* Joint first authors; \*\* Joint senior authors

### Repository layout
- `src/`: project source (e.g., `inference_functions.py`, helpers)
- `Metric Calculations/`: metrics computation and analysis utilities
- `config/settings.py`: centralised configuration reading from environment
- `data/`: place datasets here

### Requirements and installation
- Python 3.10+
- Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Configuration (.env)
This code reads configuration from environment variables via `config/settings.py`. Copy the template and fill values:
```bash
cp .env.example .env
```
Key variables:
- `DATA_ROOT`: path to the dataset root (e.g., `./data`)
- `RESULTS_ROOT`, `OUTPUT_DIR`: where results are written
- `ANNOTATIONS_GLOB`: pattern for annotation files
- `FAST_RESULTS`: set to `true` to skip slow metrics during dev
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
- `AZURE_HDS_BEARER_TOKEN`: bearer token for Azure Health DeID service
- `HF_ACCESS_TOKEN`: Hugging Face token
- Optional: `MODEL_RESULTS_MAP` JSON string overriding model->CSV mapping

### Usage
- Inference and evaluation:
  - Use the notebooks: `Inference_Pipeline.ipynb` and `src/AnonCAT_tuning.ipynb` (for tuning). Notebooks load `.env` via `config/settings.py` and expect `DATA_ROOT` and `AZURE_HDS_BEARER_TOKEN` to be set.
  - Metrics aggregation: `Metric Calculations/2025-02_batch_metrics_script.py` (now reads paths and flags from `config/settings.py`).
- New code should live under `src/`. `config/settings.py` adds `src/` to `sys.path` for imports.

### Reproducibility notes
- Data files are not tracked; place them under `data/` per your `.env` configuration.
- Secrets must never be hard-coded; keep them in `.env`.
- Azure Health DeID URL can be edited at the top of the pipeline notebook as `AZURE_HDS_URL`.

### Contributions
- **RK (Rachel Kuo)**: Lead for data labelling; contribution to metric calculation and charting; co-designed analyses; co-authored first draft of manuscript
- **AS (Andrew A.S. Soltan)**: LLM inference and fine-tuning pipelines; model infrastructure and deployment; metric calculation scripts and charting; co-designed analyses; co-authored first draft of manuscript
- **COH (Ciaran O’Hanlon), AH (Alan Hasanic)**: Contributions to data labelling.
- **GC (Gary Collins)**: Contributions to data interpretation; review of manuscript
- **DF (Dominic Furniss), DWE (David W. Eyre)**: Co-supervisors of the research.

### Citation
If you use `andrewsoltan/ehr-deidentification-for-health`, please cite: [Kuo R, Soltan AAS, O’Hanlon C, Hasanic A, Clifton DA, Gary C, Furniss D, Eyre DW. Benchmarking transformer-based models for medical record deidentification: A single centre, multi-specialty evaluation. medRxiv. 2025 May 6:2025-05.] (https://www.medrxiv.org/content/10.1101/2025.05.05.25326979v1)



