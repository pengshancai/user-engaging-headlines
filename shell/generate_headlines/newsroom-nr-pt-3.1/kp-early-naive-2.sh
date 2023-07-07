CUDA_VISIBLE_DEVICES=2 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.1/ \
    --kp_select_method early-naive \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/ \
    --output_file kp-early-naive-2.json \
    --top_k 2
