cd ../..
python scripts/data_process/prepare_general/generate_synthesized_users.py \
    --id2text_kps_file ../recsum_/data/newsroom/kp_7.0/test-id2textkps.json \
    --id2title_kps_file ../recsum_/data/newsroom/synthesized_user/test-id2titlekps.json \
    --data_file ../recsum_/data/newsroom/test.jsonl \
    --output_dir ../recsum_/data/newsroom/synthesized_user/ \
    --output_test_file ../recsum_/data/newsroom/synthesized_user/test_additional_1.json \
    --num_kps_range 10,15,20 \
    --min_num_news 30 \
    --max_num_news 35 \
    --num_synthesized_users_per_num_kps 2000 \
    --dataset nr \
    --tasks 1/2/3

python scripts/data_process/prepare_general/generate_synthesized_users.py \
    --id2text_kps_file ../recsum_/data/newsroom/kp_7.0/test-id2textkps.json \
    --id2title_kps_file ../recsum_/data/newsroom/synthesized_user/test-id2titlekps.json \
    --data_file ../recsum_/data/newsroom/test.jsonl \
    --output_dir ../recsum_/data/newsroom/synthesized_user/ \
    --output_test_file ../recsum_/data/newsroom/synthesized_user/test_additional_2.json \
    --num_kps_range 10,20,30 \
    --min_num_news 50 \
    --max_num_news 60 \
    --num_synthesized_users_per_num_kps 2000 \
    --dataset nr \
    --tasks 1/2/3
