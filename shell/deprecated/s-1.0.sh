CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --exp_id s-1.0 \
    --stage pretune \
    --model_name_or_path facebook/bart-base \
    --train_file ../recsum_/data/newsroom/train.json \
    --validation_file ../recsum_/data/newsroom/dev.json \
    --processed_dataset_file ../recsum_/data/newsroom/processed.json \
    --world_user_emb_file ../recsum_/data/processed/world_user_emb.pt \
    --text_column text \
    --summary_column title \
    --recommender_type bart-base-encoder \
    --recommender_path ../PLMNR_/dump/t-1.7.1/epoch-3-30000.pt \
    --source_prefix '<u> ' \
    --checkpointing_steps 10000 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ../recsum_/dump/s-1.0/ \
    2>&1 | tee ../recsum_/logs/s-1.0.txt