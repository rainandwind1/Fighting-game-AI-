import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np

f = open("E:/Grade_two/作业/强化学习/作业二/FTG4.50_rlhomework/info/Hp_diff_AC_ep.txt", "r")  # 设置文件对象
line = f.readline()
line = line[:-1]
count = 0
loss = []
score = []

Hp_loss = []
win_prob = []
hp_avg = []
win_count = 0
count_valid = 0
hp_sum = 0.


def plot_curse(target_list, X_label, Y_label, title, smooth_list = [],legend = None):
    figure = plt.figure()
    plt.grid()
    plt.plot([i + 1 for i in range(len(target_list))], target_list, '-r')
    if smooth_list != []:
        plt.plot([i + 1 for i in range(len(target_list))], smooth_list, '-b')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    if len(target_list) < 100:
        plt.bar([i + 1 for i in range(len(target_list))], target_list, color = 'y', width = 0.6)
        for a, b in zip([i + 1 for i in range(len(target_list))], target_list):
            plt.text(a, b + 0.05, '%.3f' % b, ha='center', va='bottom', fontsize=9)
    plt.title(title)
    plt.show()


while count < 2940:  # 直到读取完文件
    if count < 1795:
        count += 1
        continue
    line = f.readline()  # 读取一行文件，包括换行符
    line = line[:-1]  # 去掉换行符，也可以不去
    t = line.split(" ")
    t = list(t)
    if t != ['']:
        if float(t[3]) > 0:   # 获胜计数
            win_count += 1
        count_valid += 1
        hp_sum += float(t[3])
        if count_valid == 100:
            win_prob.append(float(win_count / count_valid))
            hp_avg.append(float(hp_sum / count_valid))
            count_valid = 0
            win_count = 0
            hp_sum = 0.
        score.append(float(t[3]))
    count += 1

v_smooth1 = scipy.signal.savgol_filter(score, 51, 1)
f.close()  # 关闭文件
plot_curse(score,  'Epoch', 'Hp diff', None, v_smooth1)
plot_curse(win_prob,  'Epoch (every 100 epo)', 'Win prob', None)
print(count_valid)
plot_curse(hp_avg,  'Epoch (every 100 epo)', 'Hp diff avg', None)


# score_list = np.load("info\Reward_PPO1110.npy")
# score_list = score_list.tolist()
# v_smooth2 = scipy.signal.savgol_filter(score_list, 101, 1)
loss_list = np.load("info\Loss_AC_ep1200.npy")
loss_list = loss_list.tolist()
# plot_curse(score_list,  'Epoch', 'Score', None, v_smooth2)
plot_curse(loss_list,  'Train step', 'Loss', None)

