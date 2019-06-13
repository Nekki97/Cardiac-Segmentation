import os
from matplotlib import pyplot as plt
import numpy as np


def read_value(paths):
    values = []
    per_pat = []
    for path in paths:
        index = path.rfind('0.')
        index2 = path.find('%_per_pat')
        for i in range(1, 5):
            if not path[index + i+1].isdigit():
                values.append(float(path[index:index + i]))
                break
        if path[index + 5].isdigit():
            values.append(float(path[index:index+5]))
        if path[index2 - 3].isdigit():
            per_pat.append(100)
        else:
            per_pat.append(int(path[index2 - 2:index2]))
    return np.array(values), per_pat

def get_mean(values, ind25, ind50, ind75, ind100):
    return [np.mean(values[ind25]), np.mean(values[ind50]), np.mean(values[ind75]), np.mean(values[ind100])]

def get_std(values, ind25, ind50, ind75, ind100):
    return [np.std(values[ind25]), np.std(values[ind50]), np.std(values[ind75]), np.std(values[ind100])]

def autolabel(rects, percent, ax, number, xpos='center'):
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}
    for i in range(len(rects)):
        for j in range(number):
            height = rects[i][j].get_height()
            ax.annotate('{}'.format(int(percent)),
                xy=(rects[i][j].get_x() + rects[i][j].get_width() / 2, height),
                xytext=(offset[xpos]*3, 3),  # use 3 points offset
                textcoords="offset points",  # in both directions
                ha=ha[xpos], va='bottom')

def get_means_stds_4(path, score):
    twenty = []
    fifty = []
    seventy = []
    hundred = []
    for root, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "25%_total_data" in root.lower() and score in filename.lower():  # check whether the file's DICOM
                twenty.append(os.path.join(root, filename))
            if "50%_total_data" in root.lower() and score in filename.lower():  # check whether the file's DICOM
                fifty.append(os.path.join(root, filename))
            if "75%_total_data" in root.lower() and score in filename.lower():  # check whether the file's DICOM
                seventy.append(os.path.join(root, filename))
            if "100%_total_data" in root.lower() and score in filename.lower():  # check whether the file's DICOM
                hundred.append(os.path.join(root, filename))

    print(len(twenty), len(fifty), len(seventy), len(hundred))

    twenty_values, twenty_perc = read_value(twenty)
    fifty_values, fifty_perc = read_value(fifty)
    seventy_values, seventy_perc = read_value(seventy)
    hundred_values, hundred_perc = read_value(hundred)
    amounts = [len(twenty_values), len(fifty_values), len(seventy_values), len(hundred_values)]

    twenty_zeros = [i for i in twenty_values if i == 0.0]
    fifty_zeros = [i for i in fifty_values if i == 0.0]
    seventy_zeros = [i for i in seventy_values if i == 0.0]
    hundred_zeros = [i for i in hundred_values if i == 0.0]
    zeros = [twenty_zeros, fifty_zeros, seventy_zeros, hundred_zeros]

    ind25 = [i for i in range(len(twenty_perc)) if twenty_perc[i] == 25]
    ind50 = [i for i in range(len(fifty_perc)) if fifty_perc[i] == 50]
    ind75 = [i for i in range(len(seventy_perc)) if seventy_perc[i] == 75]
    ind100 = [i for i in range(len(hundred_perc)) if hundred_perc[i] == 100]

    twenty_means = get_mean(twenty_values, ind25, ind50, ind75, ind100)
    fifty_means = get_mean(fifty_values, ind25, ind50, ind75, ind100)
    seventy_means = get_mean(seventy_values, ind25, ind50, ind75, ind100)
    hundred_means = get_mean(hundred_values, ind25, ind50, ind75, ind100)
    means = [twenty_means, fifty_means, seventy_means, hundred_means]

    print(len(twenty_means), len(fifty_means), len(seventy_means), len(hundred_means))

    twenty_stds = get_std(twenty_values, ind25, ind50, ind75, ind100)
    fifty_stds = get_std(fifty_values, ind25, ind50, ind75, ind100)
    seventy_stds = get_std(seventy_values, ind25, ind50, ind75, ind100)
    hundred_stds = get_std(hundred_values, ind25, ind50, ind75, ind100)
    stds = [twenty_stds, fifty_stds, seventy_stds, hundred_stds]
    return means, stds, zeros, len(ind25), amounts


def get_means_stds_1(path, key):
    twenty = []
    fifty = []
    seventy = []
    hundred = []
    for root, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if "25%_total_data" in root.lower() and 'dice' in filename.lower() and key in root.lower():  # check whether the file's DICOM
                twenty.append(os.path.join(root, filename))
            if "50%_total_data" in root.lower() and 'dice' in filename.lower() and key in root.lower():  # check whether the file's DICOM
                fifty.append(os.path.join(root, filename))
            if "75%_total_data" in root.lower() and 'dice' in filename.lower() and key in root.lower():  # check whether the file's DICOM
                seventy.append(os.path.join(root, filename))
            if "100%_total_data" in root.lower() and 'dice' in filename.lower() and key in root.lower():  # check whether the file's DICOM
                hundred.append(os.path.join(root, filename))

    print(len(twenty), len(fifty), len(seventy), len(hundred))

    twenty_values, twenty_perc = read_value(twenty)
    fifty_values, fifty_perc = read_value(fifty)
    seventy_values, seventy_perc = read_value(seventy)
    hundred_values, hundred_perc = read_value(hundred)
    amounts = [len(twenty_values), len(fifty_values), len(seventy_values), len(hundred_values)]

    twenty_zeros = [i for i in twenty_values if i == 0.0]
    fifty_zeros = [i for i in fifty_values if i == 0.0]
    seventy_zeros = [i for i in seventy_values if i == 0.0]
    hundred_zeros = [i for i in hundred_values if i == 0.0]
    zeros = [twenty_zeros, fifty_zeros, seventy_zeros, hundred_zeros]

    twenty_means = np.mean(twenty_values)
    fifty_means = np.mean(fifty_values)
    seventy_means = np.mean(seventy_values)
    hundred_means = np.mean(hundred_values)
    means = [twenty_means, fifty_means, seventy_means, hundred_means]

    twenty_stds = np.std(twenty_values)
    fifty_stds = np.std(fifty_values)
    seventy_stds = np.std(seventy_values)
    hundred_stds = np.std(hundred_values)
    stds = [twenty_stds, fifty_stds, seventy_stds, hundred_stds]
    return means, stds, amounts


def get_4rects(means, stds):
    twenties = ax1.bar(ind-width*3/2, means[0], width, yerr=stds[0], label='25% per patient')
    fifties = ax1.bar(ind-width*1/2, means[1], width, yerr=stds[1], label='50% per patient')
    seventies = ax1.bar(ind+width*1/2, means[2], width, yerr=stds[2], label='75% per patient')
    hundreds = ax1.bar(ind+width*3/2, means[3], width, yerr=stds[3], label='100% per patient')
    return [twenties, fifties, seventies, hundreds]


def get_1rect(mean, std, deviation, label):
    return ax1.bar(ind+deviation, mean, width, yerr=std, label=label)


def get_plot(ind, names, amount):
    ax1.set_ylim(0, 1)
    ax1.set_xticks(ind)
    ax1.set_xticklabels(names)
    ax1.set_xlabel('of total patients')
    ax1.set_title('Dice score vs percentages of patients (' + str(amount[0]) + ' evaluations per bar)')
    ax1.legend()


path = 'results/new'

dice1_means, dice1_stds, dice1_amounts = get_means_stds_1(path, '2_layers')
dice2_means, dice2_stds, dice2_amounts = get_means_stds_1(path, '3_layers')
dice3_means, dice3_stds, dice3_amounts = get_means_stds_1(path, '4_layers')
dice4_means, dice4_stds, dice4_amounts = get_means_stds_1(path, '5_layers')
#means = [dice1_means, dice2_means, dice3_means, dice4_means]
#stds = [dice1_stds, dice2_stds, dice3_stds, dice4_stds]
names = ['25%', '50%', '75%', '100%']
#data = [25, 50, 75, 100, 150, 225, 300, 425]

ind = np.arange(len(dice2_means))  # the x locations for the groups
width = 0.2  # the width of the bars

fig, (ax1) = plt.subplots(1)

#twenties, fifties, seventies, hundreds = get_4rects(means, stds)

dice1_twenties, dice1_fifties, dice1_seventies, dice1_hundreds = get_1rect(dice1_means, dice1_stds, -width*3/2, '2 layers')
dice2_twenties, dice2_fifties, dice2_seventies, dice2_hundreds = get_1rect(dice2_means, dice2_stds, -width/2, '3 layers')
dice3_twenties, dice3_fifties, dice3_seventies, dice3_hundreds = get_1rect(dice3_means, dice3_stds, width/2, '4 layers')
dice4_twenties, dice4_fifties, dice4_seventies, dice4_hundreds = get_1rect(dice4_means, dice4_stds, width*3/2, '5 layers')

get_plot(ind, names, dice3_amounts)

#autolabel(dice1_twenties)

fig.tight_layout()
plt.show()

def get_plot(path):
    means, stds, evals = get_means_stds_4(path, 'dice')
    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(means))
    width = 0.2

    ax.bar(ind - width * 3 / 2, means[0], width, yerr=stds[0], label='1')
    ax.bar(ind - width * 1 / 2, means[1], width, yerr=stds[1], label='2')
    ax.bar(ind + width * 1 / 2, means[2], width, yerr=stds[2], label='3')
    ax.bar(ind + width * 3 / 2, means[3], width, yerr=stds[3], label='4')

    names = ['25%', '50%', '75%', '100%']
    ax.set_ylim(0, 1)
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')
    ax.set_title('Dice score vs percentages of patients (' + str(evals[0]) + ' evaluations per bar)')
    ax.legend()
    fig.tight_layout()
    plt.show()
