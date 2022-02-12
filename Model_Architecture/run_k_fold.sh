folds=(0 1 2 3 4 5 6 7 8 9)
model="pase_mlp"
for fold in "${folds[@]}"; do
    echo "fold: $fold trainig..."
    python train.py --model $model --name "fold_${fold}" --do_kfold True --kfold_splits 10 --fold_index $fold 
    echo "fold: $fold testing..."
    python test.py --model $model --do_kfold True  --fold_index $fold --load "weights/pase_mlp/fold_${fold}/${model}_best.pth"
done
