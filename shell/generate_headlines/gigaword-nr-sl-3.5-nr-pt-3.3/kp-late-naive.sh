CUDA_VISIBLE_DEVICES=6 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.0/ \
    --kp_select_method late-naive \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-nr-pt-3.3/ \
    --output_file kp-late-naive.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=6 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword-nr-pt-3.3/kp-late-naive.json \
    --output_dir ../recsum_/results/gigaword-nr-pt-3.3/scores/ \
    --batch_size 32
