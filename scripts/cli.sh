#!/usr/bin/env bash
set -euo pipefail

# Lightweight CLI wrapper for common flows:
# usage: ./scripts/cli.sh <command> [--quick]
CMD=${1:-help}
QUICK=false
for arg in "$@"; do
  if [ "$arg" = "--quick" ]; then QUICK=true; fi
done

case "$CMD" in
  build)
    echo "Building vocab..."
    python build_vocab.py "${2:-data/train.txt}"
    ;;
  train)
    echo "Starting train..."
    if [ "$QUICK" = true ]; then
      python train.py --model_name "${3:-shk-turbo-mini}" --train_data "${2:-data/train.txt}" --valid_data "data/valid.txt" --learning_rate 1e-4 --train_steps 10 --warmup_steps 1 --checkpoint_interval 5 --log_dir logs/train
    else
      python train.py --model_name "${3:-shk-turbo-mini}" --train_data "${2:-data/train.txt}" --valid_data "data/valid.txt" --learning_rate 1e-4 --train_steps 10000 --warmup_steps 1000 --checkpoint_interval 1000 --log_dir logs/train
    fi
    ;;
  finetune)
    echo "Starting fine-tune..."
    python fine_tune.py --model_name "${2:-shk-turbo-mini}" --checkpoint_path "${3:-shk-turbo-mini/checkpoints/step_1000.pt}" --train_data "data/vobr_train.txt" --valid_data "data/vobr_valid.txt" --learning_rate 5e-5 --train_steps 1000 --checkpoint_interval 200 --log_dir logs/fine_tune
    ;;
  smoke)
    echo "Running quick smoke (build -> train 2 steps)"
    MODEL_NAME="shk-turbo-mini"
    mkdir -p data logs "$MODEL_NAME"
    echo "This is a CI smoke sentence." > data/train.txt
    echo "This is a CI smoke sentence." > data/valid.txt
    python build_vocab.py data/train.txt
    VOCAB_SIZE=$(python -c "import json; print(json.load(open('vocab.json'))['vocab_size'])")
    cat > "$MODEL_NAME/config.json" <<EOF
{
  "vocab_size": $VOCAB_SIZE,
  "hidden_size": 128,
  "num_layers": 2,
  "num_heads": 4,
  "batch_size": 2,
  "seq_len": 64,
  "dropout": 0.1,
  "activation_function": "gelu"
}
EOF
    python train.py --model_name "$MODEL_NAME" --train_data data/train.txt --valid_data data/valid.txt --learning_rate 1e-4 --train_steps 2 --warmup_steps 1 --checkpoint_interval 2 --log_dir logs/train
    ;;
  help|*)
    cat <<EOF
Usage: $0 <command> [args] [--quick]
Commands:
  build <train_data>
  train <train_data> <model_name> [--quick]
  finetune <model_name> <checkpoint_path>
  smoke         # quick end-to-end smoke test
EOF
    ;;
esac