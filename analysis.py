import os
from matplotlib import pyplot as plt
import numpy as np


def findpath(type, path):
    epochs = 0
    while not os.path.exists(path + '/' + str(epochs) + type):
        if epochs > 600:
            print("No data found")
            print(path)
            return False, None

        else:
            epochs += 1
    return True, path + '/' + str(epochs) + type

def get_prediction(path):
    predictions = []
    pred_bool, pred_path = findpath('epochs_mask_prediction.npy', path)
    if pred_bool:
        predictions = np.load(pred_path)
    else:
        print("DATA ERROR at", path)
    return np.array(predictions)


def get_medians_stds(path, params, params2, special, score, splits, dataset):
    def read_value(path):
        if score == 'dice':
            index = path.find('_roundeddice')
            count = 0
            for i in range(1,5):
                if path[index - i].isdigit():
                    count += 1
                else:
                    value = float(path[index-count-2:index])
                    break
        elif score == 'hd':
            index = path.find('_roundedhd')
            count = 0
            for i in range(1, 4):
                if path[index - i].isdigit():
                    count += 1
                else:
                    count2 = 0
                    for j in range(1,4):
                        if path[index - count - 1 - j].isdigit():
                            count2 += 1
                        else:
                            value = float(path[index - count - count2 - 1:index])
                            break
        return value

    def seeds(root, amount):
        seed_bool = False
        for i in range(1,amount+1):
            if ("seed_" + str(i) + "-") in root:
                seed_bool = True
        return seed_bool

    def new(filename, param, type):
        if type == "nii": return True
        if type == "pgm" and param != "5_levels": return "new" in filename
        else: return True

    data = [[[], [], [], []],
            [[], [], [], []],
            [[], [], [], []],
            [[], [], [], []]]

    for i in range(len(params)):
        for j in range(len(params2)):
            for root, subdirList, fileList in os.walk(path):
                for filename in fileList:
                    if score in filename and params[i] in root:  # check whether the file's DICOM
                        if params2[j] in root and special in root and new(filename, params2[j], dataset) and seeds(root, splits) and "nan" not in filename and "training_data" not in filename:
                            #print("**********************************")
                            if "_0.0_" in filename:
                                predictions = get_prediction(root)
                                minima = []
                                maxima = []
                                for l in range(predictions.shape[1]):
                                    minima.append(min(predictions[0, l, :]))
                                    maxima.append(max(predictions[0, l, :]))
                                if (max(maxima)-min(minima) >= 0.1): # only let's non-gray images through
                                    #print(root, filename)
                                    #print("Zero Added")
                                    data[i][j].append(os.path.join(root, filename))
                                else:
                                    #print(root, filename)
                                    #print("GRAYYYYY")
                                    #data[i][j].append(os.path.join(root, filename))
                                    continue
                            else:
                                #print(root, filename)
                                #print("Added normal")
                                data[i][j].append(os.path.join(root, filename))

    for i in range(len(data)):
        for j in range(len(data[i])):
            print("*********************************************************************************************")
            print(i, j, params[i], params2[j])
            for k in range(len(data[i][j])):
                print(read_value(data[i][j][k]), data[i][j][k])
            values = [read_value(data[i][j][k]) for k in range(len(data[i][j]))]
            print(round(np.median(values),3), round(np.std(values),3))

    medians = np.empty([4,4])
    stds = np.empty([4,4])
    amounts = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            values = [read_value(data[i][j][k]) for k in range(len(data[i][j]))]
            medians[i,j] = np.median(values)
            stds[i,j] = np.std(values)
            amounts.append(len(values))
    return medians, stds, amounts


def get_plot():

    #dataset = "nii"
    dataset = "pgm"

    #score = 'hd'
    score = 'dice'

    splits = 4

    plot = "levels"

    #plot = "slices"
    levels = "4_levels"

    if dataset == "pgm":
        path = 'pgm_results/unaugmented'
    if dataset == "nii":
        path = 'nii_results/levels_test'

    params = ['25%_total', '50%_total', '75%_total', '100%_total']

    if plot == "levels":
        special = "100%_per_pat"
        params2 = ['2_levels', '3_levels', '4_levels', '5_levels']
    if plot == "slices":
        special = levels
        params2 = ['25%_per_pat', '50%_per_pat', '75%_per_pat', '100%_per_pat']

    medians, stds, evals = get_medians_stds(path, params, params2, special, score, splits, dataset)

    if plot == "levels":
        medians = np.transpose(medians)
        stds = np.transpose(stds)

    new_evals = []
    for i in range(len(medians)):
        for j in range(len(medians[i])):
            new_evals.append(evals[j*len(medians)+i])

    if plot == "levels":
        print("evaluations2:", evals)
    if plot == "slices":
        print("evaluations2:", new_evals)

    fig, (ax) = plt.subplots(1)
    ind = np.arange(len(medians))
    width = 0.2

    ax.bar(ind - width * 3 / 2, medians[0], width, yerr=stds[0], label=(params2[0].replace('per_pat', 'slices')).replace('_',' '))
    ax.bar(ind - width * 1 / 2, medians[1], width, yerr=stds[1], label=(params2[1].replace('per_pat', 'slices')).replace('_',' '))
    ax.bar(ind + width * 1 / 2, medians[2], width, yerr=stds[2], label=(params2[2].replace('per_pat', 'slices')).replace('_',' '))
    ax.bar(ind + width * 3 / 2, medians[3], width, yerr=stds[3], label=(params2[3].replace('per_pat', 'slices')).replace('_',' '))

    names = ['25%', '50%', '75%', '100%']
    if score == 'dice':
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1, 0.1))
    elif score == 'hd':
        ax.set_ylim(0, 60)
        ax.set_yticks(np.arange(0, 60, 5))
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    ax.set_xlabel('of total patients')
    if score == "dice":
        ax.set_ylabel('Dice Score')
        ax.set_title('Dice score vs percentages of patients ('+ (special.replace("per_pat", "slices")).replace("_", " ") + ")")
    if score == "hd":
        ax.set_ylabel('Hausdorff Distance')
        ax.set_title('Hausdorff Distance vs percentage of patients ('+ (special.replace("per_pat", "slices")).replace("_", " ") + ")")
    ax.set_axisbelow(True)
    plt.grid(axis='y')
    ax.legend()
    fig.tight_layout()
    plt.show()

    plt.savefig("plots/final/")

get_plot()
