CUDA_VISIBLE_DEVICES=2 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_100.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/kp-early-ft_100.json \
    --output_dir ../recsum_/results/newsroom-nr-sl-3.5-text-davinci-003/scores/ \
    --batch_size 32
