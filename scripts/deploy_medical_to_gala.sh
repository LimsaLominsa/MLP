#!/bin/bash
# Deploy medical interpretability pipeline to Gala and submit jobs.
# Run this from your LOCAL machine.

set -e

REMOTE="gala1"
REMOTE_DIR="/mnt/raid0sata2/jingci/mlp"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "Step 1: Sync code, configs, and adapters"
echo "=========================================="

# New pipeline scripts
scp "$LOCAL_DIR/src/g104_pipeline/pubmed_signals.py" \
    "$LOCAL_DIR/src/g104_pipeline/nfcorpus_signals.py" \
    "$LOCAL_DIR/src/g104_pipeline/pubmed_faithfulness.py" \
    "$LOCAL_DIR/src/g104_pipeline/medical_rep_metrics.py" \
    "$REMOTE:$REMOTE_DIR/src/g104_pipeline/"

# New configs (12 files)
scp "$LOCAL_DIR/configs/pubmed_"*.yaml \
    "$LOCAL_DIR/configs/nfcorpus_"*.yaml \
    "$REMOTE:$REMOTE_DIR/configs/"

# SLURM scripts
scp "$LOCAL_DIR/scripts/slurm/run_medical_signals.slurm" \
    "$LOCAL_DIR/scripts/slurm/run_medical_rep_metrics.slurm" \
    "$LOCAL_DIR/scripts/slurm/run_medical_faithfulness.slurm" \
    "$REMOTE:$REMOTE_DIR/scripts/slurm/"

# Runner script
scp "$LOCAL_DIR/scripts/run_medical_interpretability.sh" \
    "$REMOTE:$REMOTE_DIR/scripts/"

# Medical LoRA adapters (~30-50 MB each, 8 dirs)
echo "Syncing LoRA adapters..."
rsync -avz --progress \
    "$LOCAL_DIR/MLP_组员微调_最新/models/pubmed/" \
    "$REMOTE:$REMOTE_DIR/MLP_组员微调_最新/models/pubmed/"
rsync -avz --progress \
    "$LOCAL_DIR/MLP_组员微调_最新/models/nfcorpus/" \
    "$REMOTE:$REMOTE_DIR/MLP_组员微调_最新/models/nfcorpus/"

# Data formatting scripts (auto-download from HuggingFace)
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/MLP_组员微调_最新/src/data/pubmed $REMOTE_DIR/MLP_组员微调_最新/src/data/nfcorpus"
scp "$LOCAL_DIR/MLP_组员微调_最新/src/data/pubmed/pubmed_formatting.py" \
    "$REMOTE:$REMOTE_DIR/MLP_组员微调_最新/src/data/pubmed/"
scp "$LOCAL_DIR/MLP_组员微调_最新/src/data/nfcorpus/nfcorpus_formatting.py" \
    "$REMOTE:$REMOTE_DIR/MLP_组员微调_最新/src/data/nfcorpus/"

echo ""
echo "=========================================="
echo "Step 2: Prepare data on Gala"
echo "=========================================="
echo "SSH into Gala and run:"
echo ""
echo "  ssh $REMOTE"
echo "  cd $REMOTE_DIR"
echo ""
echo "  # Activate your conda env"
echo "  conda activate <your-env>"
echo ""
echo "  # Generate PubMed data (~5 min, downloads from HuggingFace)"
echo "  python MLP_组员微调_最新/src/data/pubmed/pubmed_formatting.py --output_dir data/pubmed"
echo ""
echo "  # Generate NFCorpus data (~2 min, downloads from HuggingFace)"
echo "  python MLP_组员微调_最新/src/data/nfcorpus/nfcorpus_formatting.py --output_dir data/nfcorpus"
echo ""
echo "=========================================="
echo "Step 3: Submit SLURM jobs"
echo "=========================================="
echo "  # Check available GPUs first (don't affect others)"
echo "  squeue -u jingci"
echo "  sinfo -p gpu"
echo ""
echo "  # Phase 1: Signals extraction (12 parallel GPU jobs, ~40 min each)"
echo "  SIG_JOB=\$(sbatch --parsable --export=ALL,CONDA_ENV_NAME=<env> scripts/slurm/run_medical_signals.slurm)"
echo "  echo \"Signals job array: \$SIG_JOB\""
echo ""
echo "  # Phase 2: Rep metrics (CPU, runs after signals complete)"
echo "  REP_JOB=\$(sbatch --parsable --dependency=afterok:\$SIG_JOB --export=ALL,CONDA_ENV_NAME=<env> scripts/slurm/run_medical_rep_metrics.slurm)"
echo "  echo \"Rep metrics job: \$REP_JOB\""
echo ""
echo "  # Phase 3: Faithfulness (6 parallel GPU jobs, ~2h each)"
echo "  FAITH_JOB=\$(sbatch --parsable --export=ALL,CONDA_ENV_NAME=<env> scripts/slurm/run_medical_faithfulness.slurm)"
echo "  echo \"Faithfulness job array: \$FAITH_JOB\""
echo ""
echo "  # Monitor progress"
echo "  squeue -u jingci"
echo ""
echo "=========================================="
echo "Step 4: Copy results back (after all jobs complete)"
echo "=========================================="
echo "  # From LOCAL machine:"
echo "  rsync -avz $REMOTE:$REMOTE_DIR/outputs/pubmed_*/analysis/ $LOCAL_DIR/outputs/medical_results/pubmed/"
echo "  rsync -avz $REMOTE:$REMOTE_DIR/outputs/nfcorpus_*/analysis/ $LOCAL_DIR/outputs/medical_results/nfcorpus/"
echo ""
echo "Done! Deployment complete."
