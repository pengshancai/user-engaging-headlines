CUDA_VISIBLE_DEVICES=3 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/PENS/gigaword-NRMS-3000-plus-10.json \
    --output_dir ../recsum_/results/PENS/scores/ \
    --batch_size 32
