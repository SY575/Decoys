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

from joblib import Parallel, delayed
from docopt import docopt
def get_sa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)
    except:
        return -1

# Worker function
def select_and_evaluate_decoys(f, target, file_loc='./', output_loc='./', 
                               dataset='ALL', num_cand_dec_per_act=100, 
                               num_dec_per_act=50, max_idx_cmpd=10000):
    print("Processing: ", f)
    dec_results = [f]
    dec_results.append(dataset)
    # Read data
    data = decoy_utils.read_paired_file(file_loc+f)
    # Filter dupes and actives that are too small
    dec_results.append(len(set([d[0] for d in data])))
    tmp = [Chem.MolFromSmiles(d[0]) for d in data]
    data = [d for idx, d in enumerate(data) if tmp[idx] is not None \
            and tmp[idx].GetNumHeavyAtoms()>min_active_size]
    result = pd.DataFrame(data, columns=['act', 'dec'])
    
    decoy_smis_gen = list(set(result['dec']))
    decoy_mols_gen = [Chem.MolFromSmiles(smi) for smi in decoy_smis_gen]
    active_smis_gen = list(set(result['act']))
    active_mols_gen = [Chem.MolFromSmiles(smi) for smi in active_smis_gen]
    dataset = 'dude'
    print('Calc props for chosen decoys')
    actives_feat = decoy_utils.calc_dataset_props_dude(active_mols_gen)
    decoys_feat = decoy_utils.calc_dataset_props_dude(decoy_mols_gen)

    print('ML model performance')
    print(actives_feat.shape)
    print(decoys_feat.shape)
    dec_results.extend(list(decoy_utils.calc_xval_performance(
        actives_feat, decoys_feat, n_jobs=1)))

    print('DEKOIS paper metrics (LADS, DOE, Doppelganger score)')
    dec_results.append(decoy_utils.doe_score(actives_feat, decoys_feat))
    lads_scores = decoy_utils.lads_score_v2(active_mols_gen, decoy_mols_gen)
    dec_results.append(np.mean(lads_scores))
    dg_scores, dg_ids = decoy_utils.dg_score(active_mols_gen, decoy_mols_gen)
    dec_results.extend([np.mean(dg_scores), max(dg_scores)])
    
    print('Save decoy mols')
    print(dec_results)
    return dec_results

import argparse
if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='tmp.csv')
    parser.add_argument('--output_path', default='./eval/results/')
    parser.add_argument('--dataset_name', default='dude')
    parser.add_argument('--num_decoys_per_active', default=30)
    parser.add_argument('--min_num_candidates', default=500)
    parser.add_argument('--min_active_size', default=10)
    parser.add_argument('--max_idx_cmpd', default=10000)
    parser.add_argument('--target')
    args = parser.parse_args()
    
    target = args.target
    file_loc = args.data_path
    file_loc = f'{target}_{file_loc}'
    # times = int(args.times)
    output_loc = args.output_path
    dataset = args.dataset_name
    # num_dec_per_act = args.num_decoys_per_active * times
    num_cand_dec_per_act = args.min_num_candidates #* times
    num_dec_per_act = args.num_decoys_per_active # * times
    max_idx_cmpd = int(args.max_idx_cmpd)
    min_active_size = int(args.min_active_size)

    # Declare metric variables
    columns = ['File name', 'Dataset',
               'Orig num actives', 'Num actives', 'Num generated mols', 'Num unique gen mols',
               'AUC ROC - 1NN', 'AUC ROC - RF',
               'DOE score',
               'LADS score',
               'Doppelganger score mean', 'Doppelganger score max',
               ]

    # Populate CSV file with headers
    with open(output_loc+f'/{target}_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)

    # Select decoys and evaluate
    from select_new import run
    with open(output_loc+f'/{target}_results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print('*'*100)
        results = select_and_evaluate_decoys(
                f=file_loc, target=target, 
                file_loc=output_loc, 
                output_loc=output_loc, dataset=dataset, 
                num_cand_dec_per_act=num_cand_dec_per_act, num_dec_per_act=num_dec_per_act, 
                max_idx_cmpd=max_idx_cmpd)
        writer.writerow(results) 
        
        output_name = f'{target}_tmp_output.csv'
        run(f'./eval/results/{target}_tmp.csv', 
            './eval/results/'+output_name)
        
        print('='*100)
        results = select_and_evaluate_decoys(
                f=output_name, target=target, 
                file_loc=output_loc, 
                output_loc=output_loc, dataset=dataset, 
                num_cand_dec_per_act=1,#*times, 
                num_dec_per_act=1,#*times, 
                max_idx_cmpd=max_idx_cmpd)
        writer.writerow(results) 
