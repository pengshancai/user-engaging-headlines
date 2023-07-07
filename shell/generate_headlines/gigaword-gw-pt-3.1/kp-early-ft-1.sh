CUDA_VISIBLE_DEVICES=6 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.5/ \
    --kp_select_method early-ft \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/ \
    --output_file kp-early-ft-1.json \
    --top_k 1
