CUDA_VISIBLE_DEVICES=0 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword-gw-pt-3.3/kp-early-ft.json \
    --output_dir ../recsum_/results/gigaword-gw-pt-3.3/scores/ \
    --result_column_idx 1 \
    --output_name original \
    --batch_size 32

