for lr in "" "--learned-reward"
do 
    for environment in "HalfCheetahJoints"
    do 
        for pe in "regression" "bayesopt"
        do
            cmd="python3.6 -m experiments.experiment_driver $pe --reference-env-id=${environment}Default-v0 --randomized-env-id=${environment}Randomized-v0 ${lr}"
            $cmd
        done
    done
done

# "regression" "maml" "bayesopt" "adr" "simopt"