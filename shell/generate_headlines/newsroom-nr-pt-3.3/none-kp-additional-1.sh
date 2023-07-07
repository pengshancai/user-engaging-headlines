CUDA_VISIBLE_DEVICES=3 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_1.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file none-kp_additional_1.json

CUDA_VISIBLE_DEVICES=3 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_1.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/none-kp_additional_1.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32

CUDA_VISIBLE_DEVICES=6 python generate_results.py \
    --generator_dump_dir ../recsum_/dump/nr-pt-3.0/ \
    --kp_select_method none-kp \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_2.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/ \
    --output_file none-kp_additional_2.json

CUDA_VISIBLE_DEVICES=6 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test_additional_2.json \
    --result_file ../recsum_/results/newsroom-nr-pt-3.3/none-kp_additional_2.json \
    --output_dir ../recsum_/results/newsroom-nr-pt-3.3/scores/ \
    --batch_size 32
