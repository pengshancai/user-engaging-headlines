# suggested batch size - On RTX 6000: 32 / On A100: 64

CUDA_VISIBLE_DEVICES=2,3 python run_selector.py \
    --dpr_ctx_encoder_path facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/gigaword/selector/train-selector-3.0.json \
    --validation_file ../recsum_/data/gigaword/selector/valid-selector-3.0.json \
    --data_cache_path ../recsum_/data/gigaword/caches/ \
    --max_src_length 8 \
    --max_tgt_length 32 \
    --cache_file_name cache-%s-sl_3.0 \
    --src_column kp \
    --tgt_column history \
    --learning_rate 1e-5 \
    --per_device_batch_size 32 \
    --output_dir ../recsum_/dump/gw-sl-3.0/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 2 \
    --valid_per_epoch 8 \
    --num_warmup_steps 10000 \
    --n_gpus 2

CUDA_VISIBLE_DEVICES=4,5 python run_selector.py \
    --dpr_ctx_encoder_path facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/gigaword/selector/train-selector-2.0.json \
    --validation_file ../recsum_/data/gigaword/selector/valid-selector-2.0.json \
    --data_cache_path ../recsum_/data/gigaword/caches/ \
    --cache_file_name cache-%s-sl_2.0 \
    --src_column kp \
    --tgt_column history \
    --max_src_length 8 \
    --max_tgt_length 256 \
    --learning_rate 1e-5 \
    --per_device_batch_size 24 \
    --output_dir ../recsum_/dump/gw-sl-2.0/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 3 \
    --valid_per_epoch 6 \
    --num_warmup_steps 10000 \
    --n_gpus 2
