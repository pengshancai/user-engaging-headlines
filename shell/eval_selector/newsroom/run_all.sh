CUDA_VISIBLE_DEVICES=5 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-sl/scores/ \
    --output_name kp-early-ft-nr-sl-3.5\
    --selector_dump_dir ../recsum_/dump/nr-sl-3.5/ \
    --kp_select_method early-ft

CUDA_VISIBLE_DEVICES=5 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-sl/scores/ \
    --output_name kp-early-naive \
    --selector_dump_dir ../recsum_/dump/nr-sl-3.1/ \
    --kp_select_method early-naive

CUDA_VISIBLE_DEVICES=5 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-sl/scores/ \
    --output_name kp-late-ft-nr-sl-2.1 \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.1/ \
    --kp_select_method late-ft

CUDA_VISIBLE_DEVICES=5 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-sl/scores/ \
    --output_name kp-late-naive \
    --selector_dump_dir ../recsum_/dump/nr-sl-2.1/ \
    --kp_select_method late-naive

CUDA_VISIBLE_DEVICES=5 python eval_selector.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-sl/scores/ \
    --output_name kp-random \
    --kp_select_method random


