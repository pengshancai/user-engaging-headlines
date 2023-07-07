CUDA_VISIBLE_DEVICES=4 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/gigaword/kp-late-ft-2.json \
    --output_dir ../recsum_/results/gigaword/scores/ \
    --batch_size 32

