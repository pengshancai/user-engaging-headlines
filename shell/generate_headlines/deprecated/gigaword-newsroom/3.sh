CUDA_VISIBLE_DEVICES=3 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.0/ \
    --kp_select_method early-naive \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-newsroom/ \
    --output_file kp-early-naive-3.json \
    --top_k 3

CUDA_VISIBLE_DEVICES=3 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.0/ \
    --kp_select_method early-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-newsroom/ \
    --output_file kp-early-ft-3.json \
    --top_k 3

