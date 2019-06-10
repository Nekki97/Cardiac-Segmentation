import os
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt
import pandas as pd

path = ('results/')

scoring_func = "dice"

filepaths = []
labels = []
for dirName, subdirList, fileList in os.walk(path):
    for subdirname in subdirList:
        if "seed" in subdirname.lower():  # check whether the file's DICOM
            filepaths.append(os.path.join(dirName, subdirname))
scores_dice = []
scores_m = []
for filepath in filepaths:
    filepath += '/'
    lists = os.listdir(path=filepath)
    for element in lists:
        if 'dice' in element:
            if not element[3].isdigit():
                scores_dice.append(float(element[:3]))
            elif not element[4].isdigit():
                scores_dice.append(float(element[:4]))
            else:
                scores_dice.append(float(element[:5]))
        if 'matthews' in element:
            if not element[3].isdigit():
                scores_m.append(float(element[:3]))
            elif not element[4].isdigit():
                scores_m.append(float(element[:4]))
            else:
                scores_m.append(float(element[:5]))

df_dice = pd.DataFrame({"scores": scores_dice})
df_m = pd.DataFrame({"scores": scores_m})

bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
fig, (dice_fig, m_fig) = plt.subplots(2)
#plt.xticks(bins)


dice_fig.hist(df_dice.values, bins=bins, edgecolor="k")
m_fig.hist(df_m.values, bins=bins, edgecolor="k")
dice_fig.set_ylabel("# of Experiments")
m_fig.set_ylabel("# of Experiments")
dice_fig.set_title("Dice Score")
m_fig.set_title("Matthews Coeff")

plt.show()
