import logging
import argparse
import os
import pickle
import torch
import nltk
from filelock import FileLock
from transformers import (
    MODEL_MAPPING,
    SchedulerType,
    DataCollatorForSeq2Seq,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers import AutoTokenizer
from transformers.utils import is_offline_mode
from models.general import SummarizerPreTrainNaive
from utils.data_utils import DatasetRecSumPTNaive, DatasetSumm, DataModuleRecSum
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset


with open('../recsum_/dump/nr-pt-3.1-large/args.pkl', 'rb') as f:
    args_1 = pickle.load(f)


def load_summarizer(summarizer_model_path):
    config = BartConfig.from_pretrained(summarizer_model_path, output_hidden_states=True)
    summarizer = BartForConditionalGeneration.from_pretrained(summarizer_model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_path, use_fast=True)
    return summarizer, tokenizer


def load_ckpt(ckpt_path, partition_key='module.'):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['module']
    state_dict = {k.partition(partition_key)[2]: state_dict[k] for k in state_dict.keys()}
    ckpt['state_dict'] = state_dict
    return ckpt


"""
Device
"""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

"""
Load Model
"""
summarizer_model_path = '/cephfs/data/huggingface_models/facebook/bart-large'
ckpt_base_path = '/data/home/pengshancai/workspace/recsum_/dump/nr-pt-3.1-large/lightning_logs/version_0/checkpoints/'
ckpt_name = 'epoch=3-step=15543-valid_loss=0.92309982.ckpt'

summarizer, tokenizer = load_summarizer(summarizer_model_path)
model = SummarizerPreTrainNaive(args_1, summarizer, tokenizer)
ckpt_path = ckpt_base_path + ckpt_name + '/checkpoint/mp_rank_00_model_states.pt'
ckpt = load_ckpt(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
summarizer_1 = model.summarizer
summarizer_1.to(device)
summarizer_1.eval()

"""
Load Datasets
"""
data_files_summ = {"validation": args_1.validation_file_summ}
raw_datasets_summ = load_dataset('json', data_files=data_files_summ, field='data')
datasets_summ = DatasetSumm(args_1, raw_datasets_summ["validation"], tokenizer, args_1.cache_file_name % 'validation')




idx = 300
tokenizer.decode(datasets_summ[idx][0])
tokenizer.decode(datasets_summ[idx][1])

psg = "A home in Serbia has taken in 450 homeless and mistreated dogs, allowing them to live and play together in the open air every single day.  A video of the dogs frolicking in the sunshine together at the Sasa Pejčić’s shelter in Nis, Serbia, was posted to YouTube and already has more than 12,300 views.  The dogs are housed by a charity called the Harmony Fund.  “This is no ordinary rescue centre,” the fund wrote on YouTube.  “Rather than being warehoused in cages and kennels, the dogs here play all day long. They experience joy, good food and human kindness - often for the very first time.”  Many of the dogs reportedly arrive at the sanctuary malnourished or with broken bones and are nursed back to full health by their new carers.  The fund relies on donations to feed the dogs and maintain the sanctuary.  “The cost of care for each dog here is $15 a month, yet we only have sponsors for 88 dogs,” it wrote.  The sanctuary said it is stock-piling food for winter because it refuses to reduce the food intake of each dog despite a lack of funds.  The fund welcomes both one-off donations or monthly sponsors on its website."
prompt_a = "Serbia; Dog"
prompt_b = "Open Air; Harmony Fund"
prompt_c = "Rescue Centre"
inputs_a = tokenizer(prompt_a + '</s> ' + psg, return_tensors='pt').to(device)
inputs_b = tokenizer(prompt_b + '</s> ' + psg, return_tensors='pt').to(device)
inputs_c = tokenizer(prompt_c + '</s> ' + psg, return_tensors='pt').to(device)
title_a = tokenizer.decode(summarizer_1.generate(**inputs_a, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_b = tokenizer.decode(summarizer_1.generate(**inputs_b, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_c = tokenizer.decode(summarizer_1.generate(**inputs_c, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
print('A:\t' + title_a)
print('B:\t' + title_b)
print('C:\t' + title_c)


psg = "There’s an interesting new entry in political polling: the U.S.C. Dornsife/Los Angeles Times “Daybreak” poll. It’s different from other surveys because it’s a panel, which means it recontacts the same voters over and over. In 2012, a similar panel study done by RAND was considered a big success.  But so far, the U.S.C./LAT panel has consistently been far out of step with other surveys. Donald Trump has led in nearly every survey it has conducted in the last few months, by as much as seven percentage points. Even today, Hillary Clinton has only a one-point lead — even as she claims a comfortable lead nationwide. It was enough for the Drudge Report to feature the poll result prominently.  One factor that could be contributing to the panel’s tilt toward Mr. Trump is its decision to weight its sample according to how people say they voted in 2012.  The pollsters ask respondents whether they voted for President Obama or Mitt Romney. They then weight the sample so that Obama voters represent 27 percent of the panel and Romney voters represent 25 percent, reflecting the split of 51 percent to 47 percent between the two among actual voters in 2012. (The rest include newly eligible voters and those who stayed home.)  This is a seemingly straightforward choice. After all, why wouldn’t you want the poll to include the right number of voters for Mr. Obama and Mr. Romney? But very few high-quality public surveys — in fact, none that I’m aware of — regularly use self-reported past voting to adjust their samples.  There’s a very good reason: People just don’t seem to report their past vote very accurately. Answers tend to wind up biased toward the winner; often, people who vote for the loser say they “can’t remember” or say they voted for someone else.  That tendency is worth keeping in mind as the year’s inevitable fights about polling methodology get underway.  The most recent New York Times/CBS News poll, which showed a tied race between Mr. Trump and Mrs. Clinton heading into the Republican convention, found that 33 percent said they had voted for Mr. Obama and 25 percent for Mr. Romney. The results were similar in May: 41 percent for Obama and 32 percent for Mr. Romney (the numbers are higher because it"
prompt_a = "Political polling"
prompt_b = "Donald Trump; Presidential Election"
prompt_c = "U.S.C. Dornsife/Los Angeles Times"
inputs_a = tokenizer(prompt_a + '</s> ' + psg, return_tensors='pt').to(device)
inputs_b = tokenizer(prompt_b + '</s> ' + psg, return_tensors='pt').to(device)
inputs_c = tokenizer(prompt_c + '</s> ' + psg, return_tensors='pt').to(device)
title_a = tokenizer.decode(summarizer_1.generate(**inputs_a, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_b = tokenizer.decode(summarizer_1.generate(**inputs_b, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_c = tokenizer.decode(summarizer_1.generate(**inputs_c, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
print('A:\t' + title_a)
print('B:\t' + title_b)
print('C:\t' + title_c)


psg = "TechCrunch reports the world’s largest retailer accidentally leaked that and other details when a test site for the service was accidentally made public yesterday, giving customers more insight into how Walmart plans to challenge Amazon’s online dominance.  As MONEY reported earlier, ShippingPass, previously codenamed “Tahoe,” will offer unlimited three-day delivery of eligible items purchased at walmart.com and cost $50 per year—half the price of Amazon Prime.  An FAQ posted on the testing site reveals the launch will be restricted to a limited number of areas at launch. Products eligible for ShippingPass delivery will be marked on Walmart’s website with special logo, much like how Amazon identifies items eligible for Prime shipping. According to the FAQ, three-day delivery will only be guaranteed if the order is placed by noon local time.  While not all items will be eligible for three-day shipping, the leaked site revealed some items with slower delivery times—four to six days—will also ship at no cost, and ShippingPass appears to have no minimum order. Walmart currently offers free standard shipping to all customers on orders that exceed $50."
prompt_a = "Walmart Stores Inc; Amazon.com Inc"
prompt_b = "ShippingPass; Tahoe"
prompt_c = "TechCrunch; Three-day delivery"
inputs_a = tokenizer(prompt_a + '</s> ' + psg, return_tensors='pt').to(device)
inputs_b = tokenizer(prompt_b + '</s> ' + psg, return_tensors='pt').to(device)
inputs_c = tokenizer(prompt_c + '</s> ' + psg, return_tensors='pt').to(device)
title_a = tokenizer.decode(summarizer_1.generate(**inputs_a, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_b = tokenizer.decode(summarizer_1.generate(**inputs_b, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_c = tokenizer.decode(summarizer_1.generate(**inputs_c, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
print('A:\t' + title_a)
print('B:\t' + title_b)
print('C:\t' + title_c)


psg = "Home Depot, the world\'s largest home improvement chain, reported lower-than-expected results as its spring selling season got off to a slow start after a severe winter in many parts of the United States.  Shares dipped in pre-market trading. (Click here to get the latest quotes.)  The company posted earnings of 96 cents a share, excluding one-time items, on sales of $19.7 billion. Analysts had expected the company to report earnings of 99 cents a share on $19.95 billion in revenue, according to a consensus estimate from Thomson Reuters.  Including items, the company earnings $1 a share in the first quarter.  Comparable-store sales increased 2.6 percent.  \"The first quarter was impacted by a slow start to the spring selling season. But we had solid results in non-weather impacted markets and expect our sales for the year to grow in line with the guidance we previously provided,\" said Frank Blake, chairman and CEO of Home Depot in a press release.  Home Depot, however, maintained its sales growth forecast of 4.8 percent for the year ending January.  The company raised its full-year earnings forecast to $4.42 per share from $4.38 per share. The increase reflects a 4 cents per share benefit from the sale of shares in HD Supply Holdings and Home Depot\'s share buybacks this year.  The company gets much of its business from building contractors, who are vulnerable to weather-related disruptions.  Spring is also an important time for Home Depot as households prepare their gardens and get set for the barbecue season.  The company said it intended to buy back up to $3.75 billion additional shares this year.  Rival Lowe\'s is slated to post earnings Wednesday before the bell."
prompt_a = "Severe Winter"
prompt_b = "Sales Growth; Home Depot"
prompt_c = "Share; Buy Back"
inputs_a = tokenizer(prompt_a + '</s> ' + psg, return_tensors='pt').to(device)
inputs_b = tokenizer(prompt_b + '</s> ' + psg, return_tensors='pt').to(device)
inputs_c = tokenizer(prompt_c + '</s> ' + psg, return_tensors='pt').to(device)
title_a = tokenizer.decode(summarizer_1.generate(**inputs_a, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_b = tokenizer.decode(summarizer_1.generate(**inputs_b, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
title_c = tokenizer.decode(summarizer_1.generate(**inputs_c, do_sample=True).cpu().numpy()[0], skip_special_tokens=True)
print('A:\t' + title_a)
print('B:\t' + title_b)
print('C:\t' + title_c)



