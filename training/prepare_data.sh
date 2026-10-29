cd /path/to/Movens-Zero
conda activate movens-zero

SOURCE="data/Movens-5M"
OUTPUT="data/movens_demo"
python training/prepare_data.py --source "$SOURCE" --output "$OUTPUT" --target g1_reference
