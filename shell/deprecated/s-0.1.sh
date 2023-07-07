python run_summarization.py \
    --exp_id s-0.1 \
    --model_name_or_path facebook/bart-base \
    --dataset_name cnn_dailymail \
    --dataset_config "3.0.0" \
    --output_dir ../recsum_/dump/s-0.1/ \
    2>&1 | tee ../recsum_/logs/s-0.1.txt