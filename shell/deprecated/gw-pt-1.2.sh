python run_pretrain_pl.py \
    --stage pretune \
    --model_name_or_path /cephfs/data/huggingface_models/facebook/bart-base \
    --processed_summ_train_data_file ../recsum_/data/gigaword/processed_summ-data-512-64-train.tar \
    --processed_summ_train_list_file ../recsum_/data/gigaword/processed_summ-list-512-64-train.pkl \
    --processed_user_train_file ../recsum_/data/gigaword/processed_user-pt-train.pkl \
    --processed_summ_valid_data_file ../recsum_/data/gigaword/processed_summ-data-512-64-validation.tar \
    --processed_summ_valid_list_file ../recsum_/data/gigaword/processed_summ-list-512-64-validation.pkl \
    --processed_user_valid_file ../recsum_/data/gigaword/processed_user-pt-validation.pkl \
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
    --output_dir ../recsum_/dump/gw-pt-1.2/ \
    --n_gpus 8


