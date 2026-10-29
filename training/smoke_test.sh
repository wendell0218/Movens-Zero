cd /path/to/Movens-Zero
conda activate movens-zero

CUDA_VISIBLE_DEVICES=""
export CUDA_VISIBLE_DEVICES
python training/smoke_test.py
