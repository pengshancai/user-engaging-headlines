# suggested batch size - On RTX 6000: 32 / On A100: 64
# cd ../..
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python train_selector.py \
    --dpr_ctx_encoder_path facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/newsroom/selector/train-selector-3.3.json \
    --validation_file ../recsum_/data/newsroom/selector/dev-selector-3.3.json \
    --data_cache_path ../recsum_/data/newsroom/caches/ \
    --max_src_length 16 \
    --max_tgt_length 48 \
    --cache_file_name cache-%s-sl_3.3 \
    --src_column kp \
    --tgt_column history \
    --learning_rate 5e-5 \
    --per_device_batch_size 96 \
    --output_dir ../recsum_/dump/nr-sl-3.5/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 15 \
    --valid_per_epoch 2 \
    --num_warmup_steps 10000 \
    --n_gpus 6



