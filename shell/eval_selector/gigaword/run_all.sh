CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-sl/scores/ \
    --output_name kp-early-ft-gw-sl-3.5 \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.5/ \
    --kp_select_method early-ft

CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-sl/scores/ \
    --output_name kp-early-naive \
    --selector_dump_dir ../recsum_/dump/gw-sl-3.0/ \
    --kp_select_method early-naive

CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-sl/scores/ \
    --output_name kp-late-ft-gw-sl-2.0 \
    --selector_dump_dir ../recsum_/dump/gw-sl-2.0/ \
    --kp_select_method late-ft

CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-sl/scores/ \
    --output_name kp-late-naive \
    --selector_dump_dir ../recsum_/dump/gw-sl-2.0/ \
    --kp_select_method late-naive

CUDA_VISIBLE_DEVICES=3 python eval_selector.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --output_dir ../recsum_/results/gigaword-sl/scores/ \
    --output_name kp-random \
    --kp_select_method random



