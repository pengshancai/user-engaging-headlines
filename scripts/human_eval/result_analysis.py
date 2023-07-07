import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pat_score = re.compile('Headline ([0-9])\n([0-2])\n([0-2])\n([0-2])\n')


def get_single_result(path):
    with open(path) as f:
        con = f.read()
    scores_ul = defaultdict(list)
    scores_ha = defaultdict(list)
    scores_tq = defaultdict(list)
    recs = pat_score.findall(con)
    for idx, score_ul, score_ha, score_tq in recs:
        scores_ul[int(idx)].append(int(score_ul))
        scores_ha[int(idx)].append(int(score_ha))
        scores_tq[int(idx)].append(int(score_tq))
    return scores_ul, scores_ha, scores_tq


path = '../recsum_/results/human/ocean.txt'
scores_ul, scores_ha, scores_tq = get_single_result(path)
for idx in range(1, 6):
    print('Headline %s:\t%s\t%s\t%s' % (idx,
                                        round(np.mean(scores_ul[idx]), 2),
                                        round(np.mean(scores_ha[idx]), 2),
                                        round(np.mean(scores_tq[idx]), 2)))



dir = '../recsum_/results/human/'
import os
import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

pat_score = re.compile('Headline ([0-9])\n([0-2])\n([0-2])\n([0-2])\n')


def get_single_result(path):
    with open(path) as f:
        con = f.read()
    scores_ul = defaultdict(list)
    scores_ha = defaultdict(list)
    scores_tq = defaultdict(list)
    recs = pat_score.findall(con)
    for idx, score_ul, score_ha, score_tq in recs:
        scores_ul[int(idx)].append(int(score_ul))
        scores_ha[int(idx)].append(int(score_ha))
        scores_tq[int(idx)].append(int(score_tq))
    return scores_ul, scores_ha, scores_tq


path = '../recsum_/results/human/ocean.txt'
scores_ul, scores_ha, scores_tq = get_single_result(path)
for idx in range(1, 6):
    print('Headline %s:\t%s\t%s\t%s' % (idx,
                                        round(np.mean(scores_ul[idx]), 2),
                                        round(np.mean(scores_ha[idx]), 2),
                                        round(np.mean(scores_tq[idx]), 2)))



dir = '../recsum_/results/human/'


def show_results_all(dir):
    fnames = os.listdir(dir)
    scores_ul_all, scores_ha_all, scores_tq_all = defaultdict(list), defaultdict(list), defaultdict(list)
    for fname in fnames:
        if fname.startswith('.'):
            continue
        scores_ul, scores_ha, scores_tq = get_single_result(dir + fname)
        for idx in scores_ul.keys():
            scores_ul_all[idx].append(np.mean(scores_ul[idx]))
            scores_ha_all[idx].append(np.mean(scores_ha[idx]))
            scores_tq_all[idx].append(np.mean(scores_tq[idx]))
    for idx in range(1, 6):
        print('Headline %s:\t%s\t%s\t%s' % (idx,
                                            round(np.mean(scores_ul_all[idx]), 2),
                                            round(np.mean(scores_ha_all[idx]), 2),
                                            round(np.mean(scores_tq_all[idx]), 2)))


show_results_all(dir)

fnames = [fname for fname in os.listdir(dir) if not fname.startswith('.')]
scores_ul_all, scores_ha_all, scores_tq_all = defaultdict(list), defaultdict(list), defaultdict(list)
for fname in fnames:
    scores_ul, scores_ha, scores_tq = get_single_result(dir + fname)
    for idx in scores_ul.keys():
        scores_ul_all[idx].append(np.mean(scores_ul[idx]))
        scores_ha_all[idx].append(np.mean(scores_ha[idx]))
        scores_tq_all[idx].append(np.mean(scores_tq[idx]))

for idx in range(1, 6):
    print('Headline %s:\t%s\t%s\t%s' % (idx,
                                        round(np.mean(scores_ul_all[idx]), 2),
                                        round(np.mean(scores_ha_all[idx]), 2),
                                        round(np.mean(scores_tq_all[idx]), 2)))

size = len(scores_ul_all)
x = np.arange(size)
order = [1, 4, 5, 3, 2] # Original, Direct, TP-Random, TP-Naive, TP-Finetuned
ud = [np.mean(scores_ul_all[idx]) for idx in order]
ha = [np.mean(scores_ha_all[idx]) for idx in order]
tq = [np.mean(scores_tq_all[idx]) for idx in order]


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

font = {'family' : 'normal',
        'size'   : 8}

import matplotlib
matplotlib.rc('font', **font)

figure(figsize=(5, 2), dpi=1000)
ax = plt.subplot(111)
ax.set_ylim([0.8, 2])
ax.bar(x, tq,  width=width, label='Text quality', color='#afd2e9')
ax.bar(x + width, ha, width=width, label='Headline Appropriateness', color='#9D96B8')
ax.bar(x + 2 * width, ud, width=width, label='User Adaptation', color='#9A7197')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=5, prop={'size': 7})
plt.ylabel('Score')
# my_xticks = ['Direct', 'TP-Random', 'TP-Individual-N', 'TP-Individual-F', 'Original']
methods = ['Original', 'Direct', 'SP-Random', 'SP-Individual-N', 'SP-Individual-F']
plt.xticks(list(np.array(list(range(5)))),
           methods)
plt.xticks(rotation=15)
plt.subplots_adjust(bottom=0.2)

import matplotlib
matplotlib.rc('font', **font)

figure(figsize=(5, 2), dpi=1000)
ax = plt.subplot(111)
ax.set_ylim([0.8, 2])
ax.bar(x, tq,  width=width, label='Text quality', color='#afd2e9')
ax.bar(x + width, ha, width=width, label='Headline Appropriateness', color='#9D96B8')
ax.bar(x + 2 * width, ud, width=width, label='User Adaptation', color='#9A7197')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=5, prop={'size': 7})
plt.ylabel('Score')
# my_xticks = ['Direct', 'TP-Random', 'TP-Individual-N', 'TP-Individual-F', 'Original']
methods = ['Vanilla-Human', 'Vanilla-System', 'SP-Random', 'SP-Individual-N', 'SP-Individual-F']
plt.xticks(list(np.array(list(range(5)))),
           methods)
plt.xticks(rotation=15)
plt.subplots_adjust(bottom=0.2)
plt.savefig('../recsum_/imgs/human.png')
plt.show()



def show_results_all(dir):
    fnames = os.listdir(dir)
    scores_ul_all, scores_ha_all, scores_tq_all = defaultdict(list), defaultdict(list), defaultdict(list)
    for fname in fnames:
        if fname.startswith('.'):
            continue
        scores_ul, scores_ha, scores_tq = get_single_result(dir + fname)
        for idx in scores_ul.keys():
            scores_ul_all[idx].append(np.mean(scores_ul[idx]))
            scores_ha_all[idx].append(np.mean(scores_ha[idx]))
            scores_tq_all[idx].append(np.mean(scores_tq[idx]))
    for idx in range(1, 6):
        print('Headline %s:\t%s\t%s\t%s' % (idx,
                                            round(np.mean(scores_ul_all[idx]), 2),
                                            round(np.mean(scores_ha_all[idx]), 2),
                                            round(np.mean(scores_tq_all[idx]), 2)))


show_results_all(dir)

fnames = [fname for fname in os.listdir(dir) if not fname.startswith('.')]
scores_ul_all, scores_ha_all, scores_tq_all = defaultdict(list), defaultdict(list), defaultdict(list)
for fname in fnames:
    scores_ul, scores_ha, scores_tq = get_single_result(dir + fname)
    for idx in scores_ul.keys():
        scores_ul_all[idx].append(np.mean(scores_ul[idx]))
        scores_ha_all[idx].append(np.mean(scores_ha[idx]))
        scores_tq_all[idx].append(np.mean(scores_tq[idx]))

for idx in range(1, 6):
    print('Headline %s:\t%s\t%s\t%s' % (idx,
                                        round(np.mean(scores_ul_all[idx]), 2),
                                        round(np.mean(scores_ha_all[idx]), 2),
                                        round(np.mean(scores_tq_all[idx]), 2)))

size = len(scores_ul_all)
x = np.arange(size)
order = [1, 4, 5, 3, 2] # Original, Direct, TP-Random, TP-Naive, TP-Finetuned
ud = [np.mean(scores_ul_all[idx]) for idx in order]
ha = [np.mean(scores_ha_all[idx]) for idx in order]
tq = [np.mean(scores_tq_all[idx]) for idx in order]


total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

font = {'family' : 'normal',
        'size'   : 8}

import matplotlib
matplotlib.rc('font', **font)

figure(figsize=(5, 2), dpi=1000)
ax = plt.subplot(111)
ax.set_ylim([0.8, 2])
ax.bar(x, tq,  width=width, label='Text quality', color='#afd2e9')
ax.bar(x + width, ha, width=width, label='Headline Appropriateness', color='#9D96B8')
ax.bar(x + 2 * width, ud, width=width, label='User Adaptation', color='#9A7197')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=5, prop={'size': 7})
plt.ylabel('Score')
# my_xticks = ['Direct', 'TP-Random', 'TP-Individual-N', 'TP-Individual-F', 'Original']
methods = ['Original', 'Direct', 'SP-Random', 'SP-Individual-N', 'SP-Individual-F']
plt.xticks(list(np.array(list(range(5)))),
           methods)
plt.xticks(rotation=15)
plt.subplots_adjust(bottom=0.2)

import matplotlib
matplotlib.rc('font', **font)

figure(figsize=(5, 2), dpi=1000)
ax = plt.subplot(111)
ax.set_ylim([0.8, 2])
ax.bar(x, tq,  width=width, label='Text quality', color='#afd2e9')
ax.bar(x + width, ha, width=width, label='Headline Appropriateness', color='#9D96B8')
ax.bar(x + 2 * width, ud, width=width, label='User Adaptation', color='#9A7197')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True, shadow=True, ncol=5, prop={'size': 7})
plt.ylabel('Score')
# my_xticks = ['Direct', 'TP-Random', 'TP-Individual-N', 'TP-Individual-F', 'Original']
methods = ['Vanilla-Human', 'Vanilla-System', 'SP-Random', 'SP-Individual-N', 'SP-Individual-F']
plt.xticks(list(np.array(list(range(5)))),
           methods)
plt.xticks(rotation=15)
plt.subplots_adjust(bottom=0.2)
plt.savefig('../recsum_/imgs/human.png')
plt.show()




# Horizontal bar chart
# height = 0.8
# fig, ax = plt.subplots(3)
# methods = ['Original', 'Direct', 'TP-Random', 'TP-Individual-N', 'TP-Individual-F']
# y_pos = np.arange(len(methods))
# x_min = 0
# x_max = 2.0
#
# ax[0].barh(y_pos, tq, height=height, align='center', color='#52489c')
# ax[0].set_yticks(y_pos, labels=methods)
# ax[0].invert_yaxis()
# ax[0].set_xlim([x_min, x_max])
# # ax[0].set_xlabel('Text Quality')
# ax[0].set_title('Text Quality')
#
# ax[1].barh(y_pos, ha, height=height, align='center', color='#4062bb')
# ax[1].set_yticks(y_pos, labels=methods)
# ax[1].invert_yaxis()
# ax[1].set_xlim([x_min, x_max])
# # ax[1].set_xlabel('Headline Appropriateness')
# ax[1].set_title('Headline Appropriateness')
#
# ax[2].barh(y_pos, ud, height=height, align='center', color='#59c3c3')
# ax[2].set_yticks(y_pos, labels=methods)
# ax[2].invert_yaxis()
# ax[2].margins(0.05)
# ax[2].set_xlim([x_min, x_max])
# # ax[2].set_xlabel('User Adaptation')
# ax[2].set_title('User Adaptation')
#
# fig.tight_layout(pad=0.3)
# plt.savefig('../recsum_/imgs/human_.png')
# plt.show()

