function join_by {
# src: https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

folds=(0 1 2 3 4 5 6 7 8 9)
seeds=(1234)
model="ssast"

if [ ! -f "result/${model}/model_results.csv" ]; then
    echo "seed, fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8, fold_9, fold_10" > "result/${model}/model_results.csv"
fi

for seed in ${seeds[@]}; do
    exp_results=($seed)
    for fold in "${folds[@]}"; do
        # echo "fold: $fold trainig..."
        # python train.py --model $model --name "fold_${fold}" --do_kfold True --kfold_splits 10 --fold_index $fold --seed $seed
        echo "fold: $fold testing..."
        python test.py --model $model --do_kfold True  --fold_index $fold --load "weights/${model}/fold_${fold}/${model}_best.pth" --outdir "result/${model}/fold_${fold}"

        # read ouptut file store fold results to array
        exec < "result/${model}/fold_${fold}/details.txt"
        while read line; do
            if [[ $line == *"correlation"* ]]; then
                # extact the content inside the parathese
                str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
                # seperate two strings by comma, and extract numeric value after =
                correlation=$(echo $str | cut -d , -f 1 | cut -d = -f 2)
                exp_results+=("$correlation")
            fi
        done
    done
    # save all fold results to file
    echo $(join_by , "${exp_results[@]}") >> "result/${model}/model_results.csv"
done
