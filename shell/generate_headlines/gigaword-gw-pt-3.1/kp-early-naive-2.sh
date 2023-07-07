CUDA_VISIBLE_DEVICES=0 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.0/ \
    --kp_select_method early-naive \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/ \
    --output_file kp-early-naive-2.json \
    --top_k 2

CUDA_VISIBLE_DEVICES=0 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --kp_select_method random \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/ \
    --output_file kp-random-2.json \
    --top_k 2
