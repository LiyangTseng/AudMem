source activate musicmem
# models=("h_mlp" "h_lstm" "e_crnn")
models=("h_mlp" "h_lstm")
features=("all" "chords-timbre" "chords")
lr_rates=(0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05)

# iterate through all models
for model in "${models[@]}"; do
    # iterate through all learning rates
    mkdir -p "log/${model}"

    for feature in "${features[@]}"; do
        for lr_rate in ${lr_rates[@]}; do
            log_path="log/${model}/${feature}_${lr_rate}.txt"
            echo "[train] model:  $model | learning rate: $lr_rate | features: $feature"
            python train.py --model $model --name ${feature}_${lr_rate} --features $feature > $log_path
            # ref: find last match using grep
            str=$(cat $log_path | grep "weights/$model/${feature}_${lr_rate}/$model" | tail -1)
            best_weight=${str#*@}

            echo "[test] model:  $model | learning rate: $lr_rate | weight: $best_weight | features: $feature"
            python test.py --model $model --load $best_weight --features $feature >> $log_path
            
            echo "see $log_path for more log details "
        done    
    done

done

