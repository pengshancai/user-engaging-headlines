CUDA_VISIBLE_DEVICES=3 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword/kp-early-ft-1.json \
    --output_dir ../recsum_/results/gigaword/scores/ \
    --batch_size 32
