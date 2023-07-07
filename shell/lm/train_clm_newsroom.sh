CUDA_VISIBLE_DEVICES=6 python scripts/train_title_lm/train_clm.py \
    --model_name_or_path gpt2 \
    --train_file ~/workspace/recsum_/data/newsroom/train-lm.json \
    --validation_file ~/workspace/recsum_/data/newsroom/dev-lm.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --num_train_epochs 5 \
    --output_dir ~/workspace/recsum_/dump/lm-nr/