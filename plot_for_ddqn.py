import matplotlib.pyplot as plt
import scipy
from scipy import signal
import numpy as np

f = open("E:/Grade_two/作业/强化学习/作业二/FTG4.50_rlhomework/info/Hp_diff.txt", "r")  # 设置文件对象
line = f.readline()
line = line[:-1]
count = 0
loss = []
score = []


def plot_curse(target_list, X_label, Y_label, title, smooth_list = [],legend = None):
    figure = plt.figure()
    plt.grid()
    plt.plot(range(len(target_list)), target_list, '-r')
    if smooth_list != []:
        plt.plot(range(len(smooth_list)), smooth_list, '-b')
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)
    plt.show()

while count < 2000:  # 直到读取完文件
    line = f.readline()  # 读取一行文件，包括换行符
    line = line[:-1]  # 去掉换行符，也可以不去
    # print(line)
    # line = line.replace(",", "")
    # line = line.replace("[", "")
    # line = line.replace("]","")
    t = line.split(" ")
    # print(t)
    t = list(t)
    # print(count, t)
    # loss.append(float(t[5]))
    if t != ['']:
        score.append(float(t[3]))
    count += 1

v_smooth1 = scipy.signal.savgol_filter(score, 51, 1)
f.close()  # 关闭文件
plot_curse(score,  'Epoch', 'Hp diff', None, v_smooth1)


score_list = np.load("info\Reward_DDQN660.npy")
score_list = score_list.tolist()
score_list2 = np.load("info\Reward_DDQN990.npy")
score_list2 = score_list2.tolist()
score_list += score_list2
v_smooth2 = scipy.signal.savgol_filter(score_list, 51, 1)
loss_list = np.load("info\Loss_DDQN990.npy")
loss_list2 = np.load("info\Loss_DDQN660.npy")
loss_list = loss_list.tolist()
loss_list2 = loss_list2.tolist()
loss_list += loss_list2
plot_curse(score_list,  'Epoch', 'Score', None, v_smooth2)
plot_curse(loss_list,  'Train step', 'Loss', None)

