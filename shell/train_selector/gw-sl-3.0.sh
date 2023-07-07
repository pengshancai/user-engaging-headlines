# suggested batch size - On RTX 6000: 32 / On A100: 64
python run_selector.py \
    --dpr_ctx_encoder_path /cephfs/data/huggingface_models/facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path /cephfs/data/huggingface_models/facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/gigaword/selector/train-selector-3.0.json \
    --validation_file ../recsum_/data/gigaword/selector/valid-selector-3.0.json \
    --data_cache_path ../recsum_/data/gigaword/caches/ \
    --max_src_length 16 \
    --max_tgt_length 64 \
    --cache_file_name cache-%s-sl_3.0 \
    --src_column kp \
    --tgt_column history \
    --learning_rate 3e-5 \
    --per_device_batch_size 64 \
    --output_dir ../recsum_/dump/gw-sl-3.0/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 10 \
    --valid_per_epoch 2 \
    --num_warmup_steps 10000 \
    --n_gpus -1
