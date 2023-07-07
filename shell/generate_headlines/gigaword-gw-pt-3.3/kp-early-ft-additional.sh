# Test set contain users who has more interests (10, 15, 20)

CUDA_VISIBLE_DEVICES=2 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/gw-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.5/ \
    --kp_select_method early-ft \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test_additional_1.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/ \
    --output_file kp-early-ft_additional_1.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=2 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test_additional_1.json \
    --result_file ../recsum_/results/gigaword-gw-pt-3.3/kp-early-ft_additional_1.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/scores/ \
    --batch_size 32
