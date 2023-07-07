CUDA_VISIBLE_DEVICES=6 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/ \
    --output_file none-kp.json
