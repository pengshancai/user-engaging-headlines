import logging
import argparse
import os
import pickle
import nltk
from filelock import FileLock
from transformers import (
    MODEL_MAPPING,
    SchedulerType,
)
from transformers.utils import is_offline_mode
from models.recommender import load_recommender
from models.summarizer import load_summarizer
from models.general import SummarizerFineTune
from utils.data_utils import DatasetRecSumFT, DatasetSumm, DatasetUser, DataCollatorForRecSum, DataModuleRecSum
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datasets import load_dataset
from utils.reward_utils import load_retriever, RetrievalScorer, RecommenderScorer
import torch

with open( '../recsum_/dump/nr-ft-1.1/args.pkl', 'rb') as f:
    args = pickle.load(f)

device = torch.device('cuda')
summarizer, tokenizer_sum = load_summarizer(args, args.summarizer_ckpt_path)
summarizer.to(device)


def filter_titles(titles):
    titles = list(set(titles))
    titles_ = []
    for title in titles:
        if len(title.split(' ')) < 3:
            continue
        if 'new york times' in title.lower():
            continue
        titles_.append(title)
    return titles_



doc = "The nautical theme has been reduced to a corny joke in British seaside architecture, but there's a dignified restraint to it here. The modernist tides of 1930s Europe washed this elegant culture palace up on our shores thanks to an enlightened patron (Earl De La Warr, mayor of Bexhill) and two émigré architects (German Eric Mendelsohn and Chechen Serge Chermayeff). The strong horizontal lines of this 1935 building are reinforced by cantilevered balconies and minimal detailing."
doc = "A 59-year old Australian teacher with dual New Zealand citizenship is to be deported from Thailand following his arrested in the country's north on suspicion of child sex offences. Peter Dundas Walbran, originally from Sydney, has been held in immigration detention cells in Bangkok since his December 9 arrest in the northeastern town of Ubon Ratchthani, 600 kilometres away, where he taught at a local international school. "
doc = "We're assuming that when the musical instrument known as the sheng was invented in China thousands of years ago, no one knew that it would one day be used to play a magically spot-on version of the Super Mario Bros. theme music. But here we are. At a recent concert in Taipei, Taiwan, a performer treated her audience to the classic video game tune, complete with all the nuanced sound effects, played entirely on the sheng."
doc = "Neither Bradford City nor Port Vale could hand their new manager a victory on the first day of the League One season in a goalless draw played out in front of a bumper 18,558 crowd at Valley Parade. But Bradford's Stuart McCall will have more positives to take from his side's performance than opposite number Bruno Ribeiro as the home side dominated large chunks of the match without being able to supply a finishing touch to their attractive football."
doc = "McNair was a regular in the first team squad under Louis van Gaal last season, while Love made two appearances under the Dutchman. Neither of the defensive duo played under Moyes, but he was aware of their presence in the youth team and hopes to persuade them to continue their development in the North East. A joint fee of around £8 million is thought to have been agreed with United and it is now up to the players to decide if they want to leave Old Trafford. "
doc = "A taxi driver has been charged with kidnapping a young woman in Sydney's east before indecently assaulting her. The 21-year-old woman got into the taxi at Bondi Junction around 1am on September 23 and told the driver to take her to Woollahra. The driver, 43, allegedly ignored this request and continued driving along Victoria Road in Bellevue Hill. The passenger was allegedly indecently assaulted as she tried to exit the taxi, before she managed to escape. The driver fled the scene. "
inputs = tokenizer_sum(doc, return_tensors="pt").to(device)
gen_seq = summarizer.generate(**inputs, num_return_sequences=64, num_beam_groups=16, diversity_penalty=0.1, num_beams=128, length_penalty=0.1).cpu()
titles = tokenizer_sum.batch_decode(gen_seq.numpy(), skip_special_tokens=True)
titles = filter_titles(titles)











