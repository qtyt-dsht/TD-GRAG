# CulturLand-Check

Open-source package for the Xi'an urban cultural land diagnosis workflow based on knowledge graphs, retrieval, and LLM-assisted reasoning.

## What Is Included

- Core pipeline code in `src/` and `run_pipeline.py`
- Sanitized configuration in `config/params.yaml`
- Prompt templates and ontology files in `config/`
- Raw and auxiliary input data in `data/`
- Processed outputs and evaluation artifacts in `artifacts/v20260225/`
- Empty `outputs/` directory for regenerated figures and tables

## What Was Removed

- Real API keys and local Neo4j credentials
- LLM cache files and call logs
- Paper drafting files, LaTeX sources, and manuscript-only assets
- Local workspace noise such as `__pycache__`

## Quick Start

1. Create a Python environment and install dependencies.
2. Edit `config/params.yaml` and fill in your own LLM settings.
3. If you want to export into Neo4j, set `neo4j.uri`, `neo4j.user`, and `neo4j.password`.
4. Run `python run_pipeline.py --dry-run` to verify the stage configuration.
5. Run `python run_pipeline.py` for the full workflow, or pass `--stage` / `--from-stage` for partial execution.

## Directory Layout

```text
.
|-- artifacts/
|   `-- v20260225/
|-- config/
|   |-- ontology.json
|   |-- params.yaml
|   `-- prompts/
|-- data/
|   |-- auxiliary/
|   |-- policy/
|   |-- raw/
|   `-- spatial/
|-- outputs/
|-- src/
|   `-- utils/
`-- run_pipeline.py
```

## Notes

- The repository keeps both raw inputs and processed artifacts so the workflow can be inspected without recomputing every stage.
- Chinese field names are preserved in the data files because they are part of the original study corpus.
- Plot rendering uses `SimHei` by default. If that font is missing on your machine, change `visualization.font_family` in `config/params.yaml`.
- Large OSM shapefiles are included because the spatial analysis depends on them.

## Data Notes

See `docs/DATASET.md` for the mapping from the original workspace folders to this GitHub-ready package.
