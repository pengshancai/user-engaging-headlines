# 25
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_finetune_pl.py \
    --summarizer_model_path /cephfs/data/huggingface_models/facebook/bart-base \
    --summarizer_ckpt_path ../recsum_/dump/nr-pt-1.3/lightning_logs/version_0/checkpoints/epoch=2-step=18137-valid_loss=1.43461967.ckpt/checkpoint/mp_rank_00_model_states.pt \
    --recommender_type bart-base-encoder \
    --recommender_model_path /cephfs/data/huggingface_models/facebook/bart-base \
    --recommender_ckpt_path ../recsum_/dump/recommender/epoch-3-30000.pt \
    --retriever_model_path /cephfs/data/huggingface_models/facebook/dpr-ctx_encoder-single-nq-base \
    --alpha_rl 0.5 \
    --alpha_mle 0.5 \
    --alpha_rl_rec 10.0 \
    --alpha_rl_rel 0.0 \
    --num_train_epochs 3 \
    --train_file_summ ../recsum_/data/newsroom/train.json \
    --validation_file_summ ../recsum_/data/newsroom/validation.json \
    --data_file_user ../recsum_/data/processed/all_user_embs-500000.pkl \
    --data_file_world_user ../recsum_/data/processed/world_user_emb.pkl \
    --data_cache_path ../recsum_/data/newsroom/ \
    --text_column text \
    --summary_column title \
    --source_prefix '<u> ' \
    --max_source_length 512 \
    --max_target_length 64 \
    --per_device_batch_size_rl 16 \
    --per_device_batch_size_mle 16 \
    --valid_per_epoch 6 \
    --num_instance_valid 6000 \
    --output_dir ../recsum_/dump/nr-ft-1.1/ \
    --n_gpus 4


