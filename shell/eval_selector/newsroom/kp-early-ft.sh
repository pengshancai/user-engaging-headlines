CUDA_VISIBLE_DEVICES=0 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --output_name kp-early-ft-sel \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.1/ \
    --kp_select_method early-ft

CUDA_VISIBLE_DEVICES=0 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --output_name kp-early-ft-sel-3.3 \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.3/ \
    --kp_select_method early-ft

CUDA_VISIBLE_DEVICES=2 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --output_name kp-early-ft-sel-3.5 \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.5/ \
    --kp_select_method early-ft


