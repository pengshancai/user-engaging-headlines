python scripts/data_process/combine_id2kps.py \
    --data_dir ../recsum_/data/gigaword/kp_1.0/ \
    --prefix train-id2textkps

python scripts/data_process/combine_id2kps.py \
    --data_dir ../recsum_/data/gigaword/kp_1.0/ \
    --prefix valid-id2textkps

python scripts/data_process/combine_id2kps.py \
    --data_dir ../recsum_/data/gigaword/kp_1.0/ \
    --prefix test-id2textkps

python scripts/data_process/combine_id2kps.py \
    --data_dir ../recsum_/data/gigaword/kp_1.0/ \
    --prefix train-id2headlinekps


