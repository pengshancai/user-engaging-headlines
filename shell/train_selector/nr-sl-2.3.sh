# suggested batch size - On RTX 6000: 24 / On A100: 48
CUDA_VISIBLE_DEVICES=0,2,3,4 python train_selector.py \
    --dpr_ctx_encoder_path facebook/dpr-ctx_encoder-single-nq-base \
    --dpr_question_encoder_path facebook/dpr-question_encoder-single-nq-base \
    --train_file ../recsum_/data/newsroom/selector/train-selector-2.3.json \
    --validation_file ../recsum_/data/newsroom/selector/dev-selector-2.3.json \
    --data_cache_path ../recsum_/data/newsroom/caches/ \
    --max_src_length 16 \
    --max_tgt_length 256 \
    --cache_file_name cache-%s-sl_2.3 \
    --src_column kp \
    --tgt_column history \
    --learning_rate 3e-5 \
    --per_device_batch_size 24 \
    --output_dir ../recsum_/dump/nr-sl-2.3/ \
    --preprocessing_num_workers 8 \
    --num_train_epochs 10 \
    --valid_per_epoch 2 \
    --num_warmup_steps 10000 \
    --n_gpus 4
