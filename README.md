# Humans Perceive Wrong Narratives from AI Reasoning Texts

**Authors:** Mosh Levy, Zohar Elyoseph, Yoav Goldberg

- arXiv: https://arxiv.org/abs/2508.16599
- Demo: https://do-you-understand-ai.com
<!-- Previous link states retained for history -->

## Analysis

- `analysis_notebooks.ipynb`: the GSM8K test.
- `folio_analysis.ipynb`: the FOLIO test and the FOLIO control.
- `data/participant_inclusion.csv`: which completed questionnaires enter the analyses, with the reason for each exclusion.
- `mixed_models/`: the main tests with participants, problems and questions as random effects (Supplementary Table 3). `python3 mixed_models/export_long_data.py` writes one row per analysed participant and question to `mixed_models/data/`, then `Rscript mixed_models/crossed_models.R` fits the models (R packages lme4 and lmerTest); its output is in `mixed_models/crossed_models_output.txt`.
