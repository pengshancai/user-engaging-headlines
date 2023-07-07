CUDA_VISIBLE_DEVICES=6 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom/none-kp.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --result_column_idx 1 \
    --output_name original \
    --batch_size 32

