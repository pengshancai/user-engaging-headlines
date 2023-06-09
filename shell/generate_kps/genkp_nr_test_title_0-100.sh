CUDA_VISIBLE_DEVICES=0 python scripts/data_process/generate_key_phrases.py \
    --batch_size 1024 \
    --extract_target title\
    --extract_split test \
    --src_max_length 64 \
    --begin_percentage 0 \
    --end_percentage 100 \
    --input_path ../recsum_/data/newsroom/ \
    --output_path ../recsum_/data/newsroom/kp_7.0/ \
    --identifier_column url \
    --corpus newsroom \
    --hg_model_name ankur310794/bart-base-keyphrase-generation-kpTimes
