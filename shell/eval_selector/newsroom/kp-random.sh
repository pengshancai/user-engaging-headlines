CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --output_name kp-random \
    --kp_select_method random

