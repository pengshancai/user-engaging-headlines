CUDA_VISIBLE_DEVICES=2 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.3/ \
    --kp_select_method gold-kp \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file gold-kp.json \
    --top_k 1

CUDA_VISIBLE_DEVICES=2 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/gold-kp.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32



