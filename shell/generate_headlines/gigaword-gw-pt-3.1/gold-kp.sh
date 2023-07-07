CUDA_VISIBLE_DEVICES=2 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.1/ \
    --kp_select_method gold-kp \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword/ \
    --output_file gold-kp.json
