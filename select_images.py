# reading the CSV file, and selecting 20 random images with good brisq score from the "Blue Autofluorescence" modality.

import numpy as np
import pandas as pd

# importing the CSV file using Pandas
files_paths=pd.read_csv("/media/pontikos_nas/Data/NikolasPontikos/IRD/dataset1_bscore_modality.csv", sep=',', header=0)

# selecting only the "Blue Autofluorescence" modality and the images with brisq score smaller than 80
autofluor=files_paths[files_paths['ExamType1_LongName']=='Blue Autofluorescence (488 nm)']
autofluor=autofluor[autofluor['brisq.score'] < 80.0]

# selecting 20 random paths (keeping random seed fixed for future reference)
np.random.seed(111)
random_paths=np.random.randint(0, high=len(autofluor['file.path']), size=5)

# picking only the paths and writing them in a file
selected_paths=files_paths.loc[random_paths , 'file.path' ]
selected_paths.to_csv('selected_paths.txt', index=False, header=False)