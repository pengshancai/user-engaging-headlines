CUDA_VISIBLE_DEVICES=4 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.0/ \
    --kp_select_method early-naive \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-nr-pt-3.3/ \
    --output_file kp-early-naive.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=4 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword-nr-pt-3.3/kp-early-naive.json \
    --output_dir ../recsum_/results/gigaword-nr-pt-3.3/scores/ \
    --batch_size 32
