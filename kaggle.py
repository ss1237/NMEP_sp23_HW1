import pandas as pd
import numpy as np

PATH = 'output/resnet18/preds.npy'

data = np.load(PATH)
data = np.argmax(data, axis=1)

df = pd.DataFrame(data)
df.index += 1
df.to_csv('submission.csv', header=['Category'], index_label='Id')