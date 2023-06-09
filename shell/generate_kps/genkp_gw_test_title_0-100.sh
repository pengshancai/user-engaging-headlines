CUDA_VISIBLE_DEVICES=5 python scripts/data_process/generate_key_phrases.py \
    --batch_size 128 \
    --extract_target headline \
    --extract_split test \
    --src_max_length 64 \
    --begin_percentage 0 \
    --end_percentage 100 \
    --input_path ../recsum_/data/gigaword/ \
    --output_path ../recsum_/data/gigaword/kp_1.0/ \
    --identifier_column id \
    --corpus gigaword \
    --hg_model_name ankur310794/bart-base-keyphrase-generation-kpTimes
