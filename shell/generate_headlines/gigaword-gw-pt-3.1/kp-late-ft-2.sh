CUDA_VISIBLE_DEVICES=0 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-2.0/ \
    --kp_select_method late-ft \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/ \
    --top_k 2 \
    --output_file kp-late-ft-2.json





