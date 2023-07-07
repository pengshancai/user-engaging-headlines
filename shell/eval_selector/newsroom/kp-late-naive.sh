CUDA_VISIBLE_DEVICES=0 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --output_name kp-late-naive \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.1/ \
    --kp_select_method late-naive

