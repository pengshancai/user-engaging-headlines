CUDA_VISIBLE_DEVICES=1 python scripts/train_title_lm/train_mlm.py \
    --model_name_or_path roberta-base \
    --train_file ~/workspace/recsum_/data/newsroom/train-lm.json \
    --validation_file ~/workspace/recsum_/data/newsroom/dev-lm.json \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ~/workspace/recsum_/dump/mlm-nr/ \
    --num_train_epochs 5