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

def autolabel(rects, score, data, ax, multiplier, xpos='center'):

    def get_index(i,j):
        if i == 0:
            if j == 0: index = 0
            if j == 1: index = 1
            if j == 2: index = 2
            if j == 3: index = 3
        if i == 1:
            if j == 0: index = 1
            if j == 1: index = 3
            if j == 2: index = 4
            if j == 3: index = 5
        if i == 2:
            if j == 0: index = 2
            if j == 1: index = 4
            if j == 2: index = 5
            if j == 3: index = 6
        if i == 3:
            if j == 0: index = 3
            if j == 1: index = 5
            if j == 2: index = 6
            if j == 3: index = 7
        return index

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for i in range(len(rects)):
        for j in range(4):
            height = rects[i][j].get_height()
            ax.annotate('{}'.format(int(data[get_index(i,j)]*multiplier)),
                        xy=(rects[i][j].get_x() + rects[i][j].get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3),  # use 3 points offset
                        textcoords="offset points",  # in both directions
                        ha=ha[xpos], va='bottom')

def get_means_stds(path, score):
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



#path = 'results/all(0.3,0.1)'

path1 = 'results/2_layers'
path2 = 'results/3_layers'
path3 = 'results/4_layers'

dice1_means, dice1_stds, dice1_zeros, dice1_total_amount, dice1_amounts = get_means_stds(path1, 'dice')
dice2_means, dice2_stds, dice2_zeros, dice2_total_amount, dice2_amounts = get_means_stds(path2, 'dice')
dice3_means, dice3_stds, dice3_zeros, dice3_total_amount, dice3_amounts = get_means_stds(path3, 'dice')
#m_means, m_stds, m_zeros, m_total_amount, m_amounts  = get_means_stds(path, 'matthews')

names = ['25%', '50%', '75%', '100%']

data = [25, 50, 75, 100, 150, 225, 300, 425]

#print(len(dice_zeros[0]), len(dice_zeros[1]), len(dice_zeros[2]), len(dice_zeros[3]),
      #len(m_zeros[0]), len(m_zeros[1]), len(m_zeros[2]), len(m_zeros[3]))

ind = np.arange(len(dice1_means[0]))  # the x locations for the groups
width = 0.25  # the width of the bars

fig, (ax1, ax2, ax3) = plt.subplots(3)

def get_rects(means, stds, type, ax):
    twenties = ax1.bar(ind-width*3/2, means[0], width, yerr=stds[0], label='25% per patient')
    fifties = ax1.bar(ind-width*1/2, means[1], width, yerr=stds[1], label='50% per patient')
    seventies = ax1.bar(ind+width*1/2, means[2], width, yerr=stds[2], label='75% per patient')
    hundreds = ax1.bar(ind+width*3/2, means[3], width, yerr=stds[3], label='100% per patient')
    return [twenties, fifties, seventies, hundreds]

#dice1_twenties, dice1_fifties, dice1_seventies, dice1_hundreds = get_rects(dice1_means, dice1_stds, 'dice', ax1)
#dice2_twenties, dice2_fifties, dice2_seventies, dice2_hundreds = get_rects(dice2_means, dice2_stds, 'dice', ax2)
#dice3_twenties, dice3_fifties, dice3_seventies, dice3_hundreds = get_rects(dice3_means, dice3_stds, 'dice', ax3)

twenties = ax1.bar(ind-width*3/2, dice1_means[0], width, yerr=dice1_stds[0], label='25% per patient')
fifties = ax1.bar(ind-width*1/2, dice1_means[1], width, yerr=dice1_stds[1], label='50% per patient')
seventies = ax1.bar(ind+width*1/2, dice1_means[2], width, yerr=dice1_stds[2], label='75% per patient')
hundreds = ax1.bar(ind+width*3/2, dice1_means[3], width, yerr=dice1_stds[3], label='100% per patient')
dice1_rects = [twenties, fifties, seventies, hundreds]

twenties2 = ax2.bar(ind-width*3/2, dice2_means[0], width, yerr=dice2_stds[0], label='25% per patient')
fifties2 = ax2.bar(ind-width*1/2, dice2_means[1], width, yerr=dice2_stds[1], label='50% per patient')
seventies2 = ax2.bar(ind+width*1/2, dice2_means[2], width, yerr=dice2_stds[2], label='75% per patient')
hundreds2 = ax2.bar(ind+width*3/2, dice2_means[3], width, yerr=dice2_stds[3], label='100% per patient')
dice2_rects = [twenties2, fifties2, seventies2, hundreds2]

twenties3 = ax3.bar(ind-width*3/2, dice3_means[0], width, yerr=dice3_stds[0], label='25% per patient')
fifties3 = ax3.bar(ind-width*1/2, dice3_means[1], width, yerr=dice3_stds[1], label='50% per patient')
seventies3 = ax3.bar(ind+width*1/2, dice3_means[2], width, yerr=dice3_stds[2], label='75% per patient')
hundreds3 = ax3.bar(ind+width*3/2, dice3_means[3], width, yerr=dice3_stds[3], label='100% per patient')
dice3_rects = [twenties3, fifties3, seventies3, hundreds3]

#m_twenties = ax2.bar(ind-width*3/2, m_means[0], width, yerr=m_stds[0], label='25% per patient')
#m_fifties = ax2.bar(ind-width*1/2, m_means[1], width, yerr=m_stds[1], label='50% per patient')
#m_seventies = ax2.bar(ind+width*1/2, m_means[2], width, yerr=m_stds[2], label='75% per patient')
#m_hundreds = ax2.bar(ind+width*3/2, m_means[3], width, yerr=m_stds[3], label='100% per patient')
#m_rects = [m_twenties, m_fifties, m_seventies, m_hundreds]

ax1.set_ylim(0,1)
ax1.set_xticks(ind)
ax1.set_xticklabels(names)
ax1.set_xlabel('of total data')
ax1.set_title('Dice Score vs Percentages of ' + str(dice1_total_amount) + ' evaluations per bar (2 layers)')
ax1.legend()

ax2.set_ylim(0,1)
ax2.set_xticks(ind)
ax2.set_xticklabels(names)
ax2.set_xlabel('of total data')
ax2.set_title('Dice Score vs Percentages of ' + str(dice2_total_amount) + ' evaluations per bar (3 layers)')

ax3.set_ylim(0,1)
ax3.set_xticks(ind)
ax3.set_xticklabels(names)
ax3.set_xlabel('of total data')
ax3.set_title('Dice Score vs Percentages of ' + str(dice3_total_amount) + ' evaluations per bar (4 layers)')

#ax2.set_ylim(0,1)
#ax2.set_xticks(ind)
#ax2.set_xticklabels(names)
#ax2.set_xlabel('of total data')
#ax2.set_title('Matthews Coeff vs Percentages ' + str(m_total_amount) +' evaluations per bar ((0.3,0.1) split, 3 layers)')
#ax2.legend()

autolabel(dice1_rects, 'dice', data, ax1, 0.6, 'right')
autolabel(dice2_rects, 'dice', data, ax2, 0.6, 'right')
autolabel(dice3_rects, 'dice', data, ax3, 0.6, 'right')

#autolabel(m_rects, 'matthews', data, ax2, 0.6, 'right')

#ax1.plot(ind-width*3/2,np.poly1d(np.polyfit(ind-width*3/2, dice_means[0], 1))(ind-width*3/2),"b-")

fig.tight_layout()
plt.show()
