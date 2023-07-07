cd ../..
python scripts/data_process/generate_synthesized_users.py \
    --id2text_kps_file ../recsum_/data/newsroom/kp_7.0/test-id2textkps.json \
    --id2title_kps_file ../recsum_/data/newsroom/kp_7.0/test-id2titlekps.json \
    --data_file ../recsum_/data/newsroom/test.jsonl \
    --output_dir ../recsum_/data/newsroom/synthesized_user/ \
    --output_test_file ../recsum_/data/newsroom/synthesized_user/test.json \
    --num_kps_range 5 \
    --num_synthesized_users_per_num_kps 2000 \
    --dataset nr \
    --tasks 1/2/3

python scripts/data_process/generate_synthesized_users.py \
    --id2text_kps_file ../recsum_/data/newsroom/kp_1.0/dev-url2textkps.json \
    --id2title_kps_file ../recsum_/data/newsroom/kp_1.0/dev-url2titlekps.json \
    --data_file ../recsum_/data/newsroom/dev.jsonl \
    --output_dir ../recsum_/data/newsroom/synthesized_user/ \
    --output_test_file ../recsum_/data/newsroom/synthesized_user/dev.json \
    --dataset nr \
    --tasks 1




