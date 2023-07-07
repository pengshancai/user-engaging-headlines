CUDA_VISIBLE_DEVICES=6 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom/gold-kp.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --batch_size 32

