CUDA_VISIBLE_DEVICES=3 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.5/ \
    --kp_select_method early-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_1.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file kp-early-ft-1_additional_1.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=3 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_1.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/kp-early-ft-1_additional_1.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32

CUDA_VISIBLE_DEVICES=5 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.5/ \
    --kp_select_method early-ft \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_2.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file kp-early-ft-1_additional_2.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=5 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_2.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/kp-early-ft-1_additional_2.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32


