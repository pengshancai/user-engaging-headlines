CUDA_VISIBLE_DEVICES=1 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/ \
    --output_file none-kp.json

CUDA_VISIBLE_DEVICES=1 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword-gw-pt-3.3/none-kp.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/scores/ \
    --batch_size 32

