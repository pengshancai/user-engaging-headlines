CUDA_VISIBLE_DEVICES=1 python run_summarization_plain.py \
    --model_name_or_path facebook/bart-base \
    --train_file ../recsum_/data/newsroom/train.json \
    --validation_file ../recsum_/data/newsroom/dev.json \
    --text_column text \
    --summary_column title \
    --checkpointing_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ../recsum_/dump/s-0.0/ \
    2>&1 | tee ../recsum_/logs/s-0.0.txt