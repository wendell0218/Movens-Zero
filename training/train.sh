cd /path/to/Movens-Zero
conda activate movens-zero

CUDA_VISIBLE_DEVICES="0"
MANIFEST="data/train.jsonl"
CHECKPOINT="checkpoints/Movens-Zero-L/model.safetensors"
export CUDA_VISIBLE_DEVICES
python training/train.py --config configs/inference_l.json --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --batch-size 4 --accumulation 4 --max-steps 30000 --save-every 1000 --keep-last 3 --output runs/movens_zero
