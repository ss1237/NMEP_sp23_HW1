import pandas as pd
import numpy as np

PATH = 'output/resnet18_bs256lr24e-4/preds.npy'

data = np.load(PATH)
data = np.argmax(data, axis=1)

df = pd.DataFrame(data)
df.index += 1
df.to_csv('submission_resnet18_bs256lr24e-4.csv', header=['Category'], index_label='Id')