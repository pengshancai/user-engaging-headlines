CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.1/ \
    --kp_select_method gold-kp \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-gigaword/ \
    --output_file gold-kp.json

CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-gigaword/ \
    --output_file none-kp.json

CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.1/ \
    --kp_select_method random \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-gigaword/ \
    --output_file kp-random-3.json \
    --top_k 3
