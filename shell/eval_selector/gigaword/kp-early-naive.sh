CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/scores/ \
    --output_name kp-early-naive \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.0/ \
    --kp_select_method early-naive

