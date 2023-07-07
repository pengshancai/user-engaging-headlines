gcc -v
scl enable devtoolset-7 bash
python run_pretrain_pl.py \
    --stage pretune \
    --model_name_or_path ../recsum_/dump/bart-base \
    --train_file_summ ../recsum_/data/gigaword/train.json \
    --validation_file_summ ../recsum_/data/gigaword/validation.json \
    --data_file_user ../recsum_/data/processed/processed_user_emb.pkl \
    --data_cache_path ../recsum_/data/gigaword/ \
    --world_user_emb_file ../recsum_/data/processed/world_user_emb.pt \
    --text_column text \
    --summary_column title \
    --recommender_type bart-base-encoder \
    --recommender_path ../recsum_/dump/recommender/model_state_dict.pt \
    --source_prefix '<u> ' \
    --max_source_length 512 \
    --max_target_length 64 \
    --checkpointing_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ../recsum_/dump/gw-pt-1.3/ \
    --preprocessing_num_workers 1 \
    --num_train_epochs 10 \
    --n_gpus 8


