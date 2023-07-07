cd ../..

python run_selector.py \
    --dpr_ctx_encoder_path /cephfs/data/huggingface_models/facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path /cephfs/data/huggingface_models/facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/newsroom/kp_1.0/train-kp-history_2.0.json \
    --validation_file ../recsum_/data/newsroom/kp_1.0/dev-kp-history_2.0.json \
    --data_cache_path ../recsum_/data/newsroom/caches/ \
    --cache_file_name cache-%s-r_2.0 \
    --src_column kp \
    --tgt_column history \
    --num_train_steps_epoch 10000 \
    --checkpointing_steps 5000 \
    --learning_rate 1e-5 \
    --per_device_batch_size 64 \
    --output_dir ../recsum_/dump/nr-sl-2.0/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 10 \
    --num_warmup_steps 100 \
    --n_gpus 8





