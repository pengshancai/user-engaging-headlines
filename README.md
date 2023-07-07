# Personalized News Summarization

This is the repo for the ACL 2023 paper [Generating User-Engaging News Headlines](https://2023.aclweb.org/calls/main_conference/)



### Environment

The virtual environment is stored in the `environment.yml`

### Data Processing


To generate key-phrases from a segment (e.g. 0%-1%) of the dataset (newsroom or gigaword), run the following command

```
python scripts/data_process/generate_key_phrases.py \
    --batch_size 128 \
    --extract_target text \
    --extract_split dev \ 
    --src_max_length 512 \
    --tgt_max_length 100 \
    --tgt_min_length 60 \ 
    --begin_percentage 0 \
    --end_percentage 10 \  
    --input_path ~/workspace/recsum_/data/newsroom/ \
    --output_path ~/workspace/recsum_/data/newsroom/kp/ \
    --identifier_column url \ 
    --corpus newsroom \ 
    --hg_model_name ankur310794/bart-base-keyphrase-generation-kpTimes 
```
This command will generate a json file `dev-id2textkps-0-10.json` under the `output_path`, where 0-10 refers to the dataset segment is from 0% to 10%.

After generating key phrases from the entire dataset, you may combine all of them into a single json file `dev-id2textkps.json`. 

We may then go on to generate synthesized users using the following command (TODO: The generation method is to be revised, so that users are aligned)

```
python scripts/data_process/generate_synthesized_users.py \
    --id2text_kps_file ~/workspace/recsum_/data/newsroom/kp/dev-url2textkps.json \
    --id2title_kps_file ~/workspace/recsum_/data/newsroom/kp/dev-url2titlekps.json \
    --data_file ~/workspace/recsum_/data/newsroom/dev.jsonl \
    --output_path ~/workspace/recsum_/data/newsroom/kp \
    --num_synthesized_users 10000
```

### Evaluation

To generate headlines on the dev/test set, run

```
python scripts/results_generation/generate_results.py \
    --dataset_file ~/workspace/recsum_/data/newsroom/kp_%s/dev-kp-history.json \
    --kp_select_method late-ft \  # 'none-kp', 'gold-kp', 'early', 'late-ft', 'late-naive', 'random'
    --top_k 3 \
    --output_path ~/workspace/recsum_/results/my_results/ \
    --output_file my_results.json 
```

To evaluate the performance of the generated headlines, run

```
python scripts/results_analysis/evaluate_generated_headlines.py \
    --eval_kp_headline_relevance \
    --eval_headline_content_relevance \
    --eval_recommendation_scores \
    --eval_factcc_scores \
    --dataset_file ~/workspace/recsum_/data/newsroom/kp_7.0/dev-kp-history_1.3.1.json \
    --results_file ~/workspace/recsum_/results/kp/nr-sl-late-2.0-top-3-1.3.1.json \
    --output_file_id my_exp \
    --output_path ~/workspace/recsum_/results/kp/ \
    
```


### Train Summarizer

To train the naive summarization model, run
```
sh shell/pre-train_summarizer/nr-pt-3.0.sh
```

To train the key-phrase based summarization model, run
```
sh shell/pre-train_summarizer/nr-pt-3.1-large.sh
```

### Train Key Phrase Selector

To train the late meet selector (One single KP meets the entire user history), run
```
sh shell/train_selector/nr-sl-2.0.sh
```
To train the early meet selector (One single KP meets a single title), run
```
sh shell/train_selector/nr-sl-3.0.sh
```










