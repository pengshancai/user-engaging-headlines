CUDA_VISIBLE_DEVICES=0 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.1/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.0/ \
    --kp_select_method late-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/ \
    --output_file kp-late-tc.json
