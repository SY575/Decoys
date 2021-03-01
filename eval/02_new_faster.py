import os, csv

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import decoy_utils

import sascorer
from pss import get_pss_from_smiles

def get_sa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)
    except:
        return -1

def get_fp(mol):
    return AllChem.GetMorganFingerprint(mol, 2)

def get_scaffold_simi(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Worker function
def select_and_evaluate_decoys(
        f, target, file_loc='./', output_loc='./', T_simi=0.15, N=5):
    print("Processing: ", f)
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
    data = pd.DataFrame(data, columns=['act', 'dec'])#.sample(frac=0.01).reset_index(drop=True)
    mol_acts = [Chem.MolFromSmiles(smi) for smi in data['act'].values]
    mol_decs = [Chem.MolFromSmiles(smi) for smi in data['dec'].values]
    fp_acts = [get_fp(mol) for mol in mol_acts]
    fp_decs = [get_fp(mol) for mol in mol_decs]
    simi = [get_scaffold_simi(fp_acts[i], fp_decs[i]) for i in range(len(fp_acts))]
    idxs = np.where(np.array(simi)<T_simi)
    mol_acts = np.array(mol_acts)[idxs]
    mol_decs = np.array(mol_decs)[idxs]
    
    pss = get_pss_from_smiles(mol_acts, mol_decs)
    data = pd.DataFrame(data.values[idxs], columns=['act', 'dec'])
    data['pss'] = pss.mean(0)
    data['score'] = data['pss']
    result = []
    for key, tmp_df in data.groupby('act'):
        tmp_df = tmp_df.sort_values('score', ascending=False)
        tmp_df = tmp_df.reset_index(drop=True)
        for i in range(min([N, tmp_df.shape[0]])):
            result.append([key, tmp_df['dec'].values[i]])
    result = pd.DataFrame(result, columns=['act', 'dec'])
    result = result.drop_duplicates().reset_index(drop=True)
    result.to_csv(f'{target}_decoys.smi', index=False, header=None, sep=' ')
    
    
import argparse
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='tmp.csv')
    parser.add_argument('--T_simi', default=0.15, type=float)
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--output_path', default='./eval/results/')
    parser.add_argument('--target')
    args = parser.parse_args()
    
    target = args.target
    file_loc = args.data_path
    file_loc = f'{target}_{file_loc}'
    # times = int(args.times)
    output_loc = args.output_path

    # Select decoys and evaluate
    results = select_and_evaluate_decoys(
            f=file_loc, target=target, 
            file_loc=output_loc, 
            output_loc=output_loc)
        
