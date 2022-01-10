source activate musicmem
models=("h_mlp" "h_lstm" "e_crnn")
lr_rates=(0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05)

# iterate through all models
for model in "${models[@]}"; do
    # iterate through all learning rates
    mkdir -p "log/${model}"

    for lr_rate in ${lr_rates[@]}; do
        
        log_path="log/${model}/${lr_rate}.txt"
        echo $log_path
        # echo "[train] model:  $model | learning rate: $lr_rate"
        # python train.py --model $model --name $lr_rate > $log_path
        # # ref: find last match using grep
        # str=$(cat $log_path | grep "weights/$model/$lr_rate/$model" | tail -1)
        # best_weight=${str#*@}

        # echo "[test] model:  $model | learning rate: $lr_rate | weight: $best_weight"
        # python test.py --model $model --load $best_weight >> $log_path
        
        # echo "log details see "
    done
done

