import os
from matplotlib import pyplot as plt
import numpy as np

def get_means_stds(path, params, params2, score):
    def read_value(paths):
        values = []
        per_pat = []
        for path in paths:
            index = path.rfind('0.')
            index2 = path.find('%_per_pat')
            for i in range(1, 5):
                if not path[index + i + 1].isdigit():
                    values.append(float(path[index:index + i]))
                    break
            if path[index + 5].isdigit():
                values.append(float(path[index:index + 5]))
            if path[index2 - 3].isdigit():
                per_pat.append(100)
            else:
                per_pat.append(int(path[index2 - 2:index2]))
        return np.array(values), per_pat

    data = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    for i in range(len(params)):
        for j in range(len(params2)):
            for root, subdirList, fileList in os.walk(path):
                for filename in fileList:
                    if score in filename.lower() and params[i] in root.lower():  # check whether the file's DICOM
                        if params2[j] in root.lower() and "" in root.lower():
                            data[i][j].append(os.path.join(root, filename))

    values = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    perc = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    means = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    stds = [[[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]]
    amounts = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            values[i][j], perc[i][j] = read_value(data[i][j])
            means[i][j] = np.mean(values[i][j])
            stds[i][j] = np.std(values[i][j])
            amounts.append(len(values[i][j]))
    return means, stds, amounts


def get_plot(path):
    params = ['25%_per_pat', '50%_per_pat', '75%_per_pat', '100%_per_pat']

    params2 = ['25%_total', '50%_total', '75%_total', '100%_total']
    means, stds, evals = get_means_stds(path, params, params2, 'dice')
    print("evaluations:", evals)

    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(means))
    width = 0.2

    ax.bar(ind - width * 3 / 2, means[0], width, yerr=stds[0], label='25% of slices')
    ax.bar(ind - width * 1 / 2, means[1], width, yerr=stds[1], label='50% of slices')
    ax.bar(ind + width * 1 / 2, means[2], width, yerr=stds[2], label='75% of slices')
    ax.bar(ind + width * 3 / 2, means[3], width, yerr=stds[3], label='100% of slices')

    names = ['25%', '50%', '75%', '100%']
    ax.set_ylim(0, 1)
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')
    ax.set_title('Dice score vs percentages of patients (' + str(max(evals)) + ' evaluations per bar)')
    ax.legend()
    fig.tight_layout()
    plt.show()

get_plot('results/new')
