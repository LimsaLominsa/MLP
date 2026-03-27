# G104 Personal Code Snapshot

This folder is a cleaned GitHub-ready snapshot of the code I contributed locally for the final G104 coursework project.

## Included

- `configs/`
  - experiment configs for the final cross-domain faithfulness setup
- `scripts/`
  - run scripts, plotting scripts, and cluster helpers
- `src/g104_pipeline/`
  - local pipeline code for training, evaluation, deletion analysis, and representation analysis
- `data_preprocessing/billsum/`
  - BillSum preprocessing scripts
- `data_preprocessing/casehold/`
  - CaseHOLD preprocessing scripts
- `requirements.txt`

## Not Included

- teammate base repository snapshot
- model weights
- results
- logs
- generated datasets

## Notes

- A few scripts assume project-relative paths such as `outputs/` and `data/`.
- Remote helper scripts still target the `gala1` workflow used in this project.
