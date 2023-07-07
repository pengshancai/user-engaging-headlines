python run_selector.py \
    --dpr_ctx_encoder_path /cephfs/data/huggingface_models/facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path /cephfs/data/huggingface_models/facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/newsroom/train-kp-history_2.0.json \
    --validation_file ../recsum_/data/newsroom/dev-kp-history_2.0.json \
    --data_cache_path ../recsum_/data/newsroom/ \
    --cache_file_name cache-%s-r \
    --checkpointing_steps 10000 \
    --learning_rate 2e-5 \
    --per_device_batch_size 64 \
    --output_dir ../recsum_/dump/nr-sl-1.0-x/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 100 \
    --n_gpus 1


#    --loss_type xent \