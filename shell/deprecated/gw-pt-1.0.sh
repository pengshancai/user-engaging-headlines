python run_pretrain_pl.py \
    --stage pretune \
    --model_name_or_path facebook/bart-base \
    --train_file ../recsum_/data/gigaword/train.jsonl \
    --validation_file ../recsum_/data/gigaword/dev.jsonl \
    --processed_dataset_file ../recsum_/data/gigaword/processed-512-64.pkl \
    --world_user_emb_file ../recsum_/data/processed/world_user_emb.pt \
    --text_column text \
    --summary_column title \
    --recommender_type bart-base-encoder \
    --recommender_path ../PLMNR_/dump/t-1.7.1/epoch-3-30000.pt \
    --source_prefix '<u> ' \
    --max_source_length 512 \
    --max_target_length 64 \
    --checkpointing_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ../recsum_/dump/gw-pt-1.0/ \
    --n_gpus 8

