# RoboCasa365 job interface

Use the repository-level [README](../../../README.md) for the only supported execution design: one task × one arm × seed-0 job per GPU, training followed immediately by held-out evaluation.

The complete 24-job assignment list is [configs/jobs_seed0.tsv](configs/jobs_seed0.tsv), and the runner is [scripts/run_task_arm.py](scripts/run_task_arm.py). Do not use separate all-arm, training-only, or evaluation-only workflows for this study.
