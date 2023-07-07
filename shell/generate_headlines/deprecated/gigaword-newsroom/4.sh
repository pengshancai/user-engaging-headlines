CUDA_VISIBLE_DEVICES=4 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-2.0/ \
    --kp_select_method late-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-newsroom/ \
    --output_file kp-late-ft.json \
    --top_k 3

CUDA_VISIBLE_DEVICES=4 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-2.0/ \
    --kp_select_method late-naive \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-newsroom/ \
    --top_k 3 \
    --output_file kp-late-naive-3.json
