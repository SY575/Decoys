target="HIVPR"
T_simi=0.15
N_decoys_per_act=5
python ./scripts/sample.py --model_load $1 --target $target --num_per_act 10 & wait
python ./eval/01_result2input.py --target $target & wait
python ./eval/02_new_faster.py --target $target --T_simi $T_simi --N $N_decoys_per_act & wait
