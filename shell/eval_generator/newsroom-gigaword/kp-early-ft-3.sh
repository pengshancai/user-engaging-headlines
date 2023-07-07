CUDA_VISIBLE_DEVICES=3 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom-gigaword/kp-early-ft-3.json \
    --output_dir ../recsum_/results/newsroom-gigaword/scores/ \
    --batch_size 32

