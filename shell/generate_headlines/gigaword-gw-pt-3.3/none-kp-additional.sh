CUDA_VISIBLE_DEVICES=2 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test_additional_1.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/ \
    --output_file none-kp_additional_1.json

CUDA_VISIBLE_DEVICES=2 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test_additional_1.json \
    --result_file ../recsum_/results/gigaword-gw-pt-3.3/none-kp_additional_1.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/scores/ \
    --batch_size 32

