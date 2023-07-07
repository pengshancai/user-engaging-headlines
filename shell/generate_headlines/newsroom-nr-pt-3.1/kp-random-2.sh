CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.1/ \
    --kp_select_method random \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/ \
    --output_file kp-random-2.json \
    --top_k 2
