# Generation based on a single KP; Large model

python train_generator.py \
    --model_name_or_path /cephfs/data/huggingface_models/facebook/bart-large \
    --train_file_summ ../recsum_/data/newsroom/train-g.json \
    --validation_file_summ ../recsum_/data/newsroom/dev-g.json \
    --data_cache_path ../recsum_/data/newsroom/ \
    --cache_file_name cache-%s-g \
    --text_column text \
    --summary_column title \
    --max_source_length 512 \
    --max_target_length 64 \
    --checkpointing_steps 10000 \
    --learning_rate 5e-5 \
    --per_device_batch_size_mle 16 \
    --output_dir ../recsum_/dump/nr-pt-3.3-large-x/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 10 \
    --n_gpus 1


