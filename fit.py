from unimol_tools import MolTrain, MolPredict
import numpy as np
import pickle

with open('data.pkl','rb') as f:
    raw = pickle.load(f)
    
data={
    'target':raw['target'],
    'atoms':raw['atoms'],
    'coordinates':raw['coordinates'],
}

clf = MolTrain(task='regression', data_type='molecule', epochs=10, batch_size=112, metrics='r2')
pred = clf.fit(data = data)

# clf = MolPredict(load_model='../exp')
# res = clf.predict(data = data)
