import json
from collections import defaultdict
import matplotlib.pyplot as plt

with open('../recsum_/data/newsroom/dev-title-key_phrases.json') as f:
    con = json.load(f)

num_kp2cnt = defaultdict(int)
for kps in con:
    num_kp = 4 if len(kps) > 4 else len(kps)
    num_kp2cnt[num_kp] += 1


with open('../recsum_/data/newsroom/kp_7.0/dev-text-key_phrases.json') as f:
    con = json.load(f)

lower_bound = 7
upper_bound = 18
num_kp2cnt = defaultdict(int)
for kps in con:
    if type(kps) == str:
        kps = kps[:-1].split(';')
    # num_kp = upper_bound if len(kps) > upper_bound else len(kps)
    # num_kp = lower_bound if len(kps) < lower_bound else len(kps)
    if len(kps) > upper_bound:
        num_kp2cnt[upper_bound] += 1
    elif len(kps) < lower_bound:
        num_kp2cnt[lower_bound] += 1
    else:
        num_kp2cnt[len(kps)] += 1


# num_kp2cnt = {29: 6740, 30: 6723, 24: 3641, 21: 2501, 27: 5887, 23: 3104, 31: 6735, 34: 5199, 25: 4447, 28: 6289, 35: 4270, 14: 985, 38: 1975, 19: 2431, 36: 3403, 26: 5050, 18: 2441, 33: 5770, 10: 1604, 37: 2606, 16: 1988, 32: 6409, 15: 1439, 17: 2233, 13: 728, 22: 2659, 20: 2483, 39: 1485, 8: 965, 9: 1577, 41: 664, 12: 856, 11: 1285, 7: 244, 6: 36, 40: 997, 42: 423, 46: 37, 44: 156, 43: 250, 45: 82, 5: 9, 48: 8, 47: 15, 49: 5, 51: 1, 50: 2}

x = [i for i in range(lower_bound, upper_bound)]
y = [num_kp2cnt[i] for i in range(lower_bound, upper_bound)]



plt.bar(x, y)
plt.ylabel("Number of passages")
plt.xlabel("Number of KPs")
# plt.title('')
plt.show()




ls = []
for kps in con:
    if type(kps) == str:
        kps = kps[:-1].split(';')
    ls.append(len(kps))






# for i in range(5, 46):
#     if i not in num_kp2cnt:
#         print(i)


