CUDA_VISIBLE_DEVICES=4 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom/kp-early-ft-4.json \
    --output_dir ../recsum_/results/newsroom/scores/ \
    --batch_size 32
