# Generation based on one single KP; Large model
CUDA_VISIBLE_DEVICES=0,1 python train_generator.py \
    --model_name_or_path facebook/bart-large \
    --train_file_summ ../recsum_/data/gigaword/kp_1.1/train.jsonl \
    --validation_file_summ ../recsum_/data/gigaword/kp_1.1/valid.jsonl \
    --data_cache_path ../recsum_/data/gigaword/kp_1.1/ \
    --cache_file_name cache-%s-k_1.1 \
    --text_column text \
    --summary_column title \
    --max_source_length 512 \
    --max_target_length 64 \
    --checkpointing_steps 50000 \
    --learning_rate 5e-5 \
    --per_device_batch_size_mle 48 \
    --output_dir ../recsum_/dump/gw-pt-3.3/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 5 \
    --n_gpus 2
