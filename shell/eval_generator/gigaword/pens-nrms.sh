CUDA_VISIBLE_DEVICES=0 python eval_generator.py \
    --dataset_file ../recsum_/data/gigaword/synthesized_user/test.json \
    --result_file ../recsum_/results/PENS/gigaword-NRMS.json \
    --output_dir ../recsum_/results/PENS/scores/ \
    --batch_size 32
