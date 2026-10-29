cd "$(dirname "${BASH_SOURCE[0]}")/.."
conda activate movens-zero
python evaluation/generate.py --track text --data-dir data/intent/text --output-dir outputs/intent/text --checkpoint checkpoints/Movens-Zero-XL/model.safetensors --model-size XL
