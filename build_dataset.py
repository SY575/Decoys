import pandas as pd
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='tmp.smi')
parser.add_argument('--target', default='TARGET_NAME')
args = parser.parse_args()
act = pd.read_csv(args.path, sep=' ',
                 header=None, names=['smi'])['smi']
act = list(set(act))

dec = pd.read_csv('./zinc_all.csv', usecols=['smiles'])
dec = dec.sample(frac=1.0).reset_index(drop=True)['smiles'].values[:len(act)*100]
dec = list(set(dec))

target = args.target # target name
pickle.dump(
    act, open(f'./MolSty-PyTorch/data/content_train_{target}.pkl', 'wb'))
pickle.dump(
    act, open(f'./MolSty-PyTorch/data/content_test_{target}.pkl', 'wb'))
pickle.dump(
    dec, open(f'./MolSty-PyTorch/data/style_instance_train_{target}.pkl', 'wb'))
pickle.dump(
    dec, open(f'./MolSty-PyTorch/data/style_instance_test_{target}.pkl', 'wb'))
