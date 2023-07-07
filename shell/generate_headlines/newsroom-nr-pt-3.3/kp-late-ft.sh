CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.1/ \
    --kp_select_method late-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file kp-late-ft-1.json \
    --top_k 1
``
CUDA_VISIBLE_DEVICES=5 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/kp-late-ft-1.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32
