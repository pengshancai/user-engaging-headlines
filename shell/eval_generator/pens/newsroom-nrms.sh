CUDA_VISIBLE_DEVICES=4 python eval_generator.py \
    --dataset_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --result_file ../recsum_/results/PENS/newsroom-NRMS-2000.json \
    --output_dir ../recsum_/results/PENS/scores/ \
    --batch_size 32
