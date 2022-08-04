function join_by {
# src: https://stackoverflow.com/questions/1527049/how-can-i-join-elements-of-an-array-in-bash
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

folds=(0 1 2 3 4 5 6 7 8 9)
seeds=(1111 1000 2345 1234)
model="e_cnn"
mkdir -p "results/$model"
correlation_results_file="results/${model}/correlation_results.csv"
loss_results_file="results/${model}/loss_results.csv"

if [ ! -f $correlation_results_file ]; then
    echo "write to new correlation results file"
    echo "seed, fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8, fold_9, fold_10" > $correlation_results_file
fi
if [ ! -f $loss_results_file ]; then
    echo "write to new loss results file"
    echo "seed, fold_1, fold_2, fold_3, fold_4, fold_5, fold_6, fold_7, fold_8, fold_9, fold_10" > $loss_results_file
fi

# vanilla (no augmentation, LDS)
echo "norm_label_vanilla" >> $correlation_results_file
echo "norm_label_vanilla" >> $loss_results_file
for seed in ${seeds[@]}; do
    correlation_results=($seed)
    loss_results=($seed)
    for fold in "${folds[@]}"; do
        exp_name="seed_${seed}_fold_${fold}_norm_label_vanilla"
        echo "trainig ${exp_name} with ${model}..."
        python train.py --model $model --name $exp_name --kfold_splits 10 --fold_index $fold --seed $seed --use_lds False
        echo "testing ${exp_name} with ${model}..."
        python test.py --model $model --fold_index $fold --load "weights/${model}/${exp_name}/${model}_best.pth" --outdir "results/${model}/${exp_name}"
        # read ouptut file store fold results to array
        exec < "results/${model}/${exp_name}/details.txt"
        while read line; do
            if [[ $line == *"correlation"* ]]; then
                # extact the content inside the parathese
                str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
                # seperate two strings by comma, and extract numeric value after =
                correlation=$(echo $str | cut -d , -f 1 | cut -d = -f 2)
                correlation_results+=("$correlation")
            fi
            if [[ $line == *"loss"* ]]; then
                # extact the content inside the parathese
                str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
                # seperate two strings by comma, and extract numeric value after =
                loss=$(echo $str | cut -d , -f 1 )
                loss_results+=("$loss")
            fi

        done
    done
    # save all fold results to file
    echo $(join_by , "${correlation_results[@]}") >> $correlation_results_file
    echo $(join_by , "${loss_results[@]}") >> $loss_results_file
done

# with augmentation (no LDS) 
echo "norm_label_w/aug" >> $correlation_results_file
echo "norm_label_w/aug" >> $loss_results_file
for seed in ${seeds[@]}; do
    correlation_results=($seed)
    loss_results=($seed)
    for fold in "${folds[@]}"; do
        exp_name="seed_${seed}_fold_${fold}_norm_label_w/aug"
        echo "trainig ${exp_name} with ${model}..."
        python train.py --model $model --name $exp_name --kfold_splits 10 --fold_index $fold --seed $seed --use_pitch_shift --use_lds False
        echo "testing ${exp_name} with ${model}..."
        python test.py --model $model --fold_index $fold --load "weights/${model}/${exp_name}/${model}_best.pth" --outdir "results/${model}/${exp_name}"
        # read ouptut file store fold results to array
        exec < "results/${model}/${exp_name}/details.txt"
        while read line; do
            if [[ $line == *"correlation"* ]]; then
                # extact the content inside the parathese
                str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
                # seperate two strings by comma, and extract numeric value after =
                correlation=$(echo $str | cut -d , -f 1 | cut -d = -f 2)
                correlation_results+=("$correlation")
            fi
            if [[ $line == *"loss"* ]]; then
                # extact the content inside the parathese
                str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
                # seperate two strings by comma, and extract numeric value after =
                loss=$(echo $str | cut -d , -f 1 )
                loss_results+=("$loss")
            fi

        done
    done
    # save all fold results to file
    echo $(join_by , "${correlation_results[@]}") >> $correlation_results_file
    echo $(join_by , "${loss_results[@]}") >> $loss_results_file
done

# with LDS (no augmentation)
# echo "norm_label_w/lds" >> $correlation_results_file
# for seed in ${seeds[@]}; do
#     correlation_results=($seed)
#     loss_results=($seed)
#     for fold in "${folds[@]}"; do
#         exp_name="seed_${seed}_fold_${fold}_norm_label_w/lds"
#         echo "trainig ${exp_name} with ${model}..."
#         python train.py --model $model --name $exp_name --kfold_splits 10 --fold_index $fold --seed $seed --use_lds True
#         echo "testing ${exp_name} with ${model}..."
#         python test.py --model $model --fold_index $fold --load "weights/${model}/${exp_name}/${model}_best.pth" --outdir "results/${model}/${exp_name}"
#         # read ouptut file store fold results to array
#         exec < "results/${model}/${exp_name}/details.txt"
#         while read line; do
#             if [[ $line == *"correlation"* ]]; then
#                 # extact the content inside the parathese
#                 str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
#                 # seperate two strings by comma, and extract numeric value after =
#                 correlation=$(echo $str | cut -d , -f 1 | cut -d = -f 2)
#                 correlation_results+=("$correlation")
#             fi
#             if [[ $line == *"loss"* ]]; then
#                 # extact the content inside the parathese
#                 str=$(echo $line | cut -d '(' -f 2 | cut -d ')' -f 1)
#                 # seperate two strings by comma, and extract numeric value after =
#                 loss=$(echo $str | cut -d , -f 1 )
#                 loss_results+=("$loss")
#             fi

#         done
#     done
#     # save all fold results to file
#     echo $(join_by , "${correlation_results[@]}") >> $correlation_results_file
#     echo $(join_by , "${loss_results[@]}") >> $loss_results_file
# done
