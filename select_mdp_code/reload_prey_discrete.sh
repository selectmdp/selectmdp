#!/bin/bash
total_reload=400
CUDA_VISIBLE=10
TRAIN_EP=100
Trainstep=175
EXPAND=6
LAYERs=4
BUFSIZE=50000
# N_CHOOSE=0
seed=0
FINAL_EPS=0.1
Env="predator_prey_discrete"
GRID_SIZE=10
UNSELECT=4
NET="SHAREEINET"
NET="DIFFEINET"
gpu_config=0.093
TARGET_UPDATE=1000
start_reload=400
end_reload=800
# AGENT="dqn_ind"
AGENT="CENTRAL_GREEDY"
for N_CHOOSE in 10
do
    for BAT_SIZE in 32
    do
        for n_test in 10
        do
            for LR in 0.001
            do
                for seed in 0 1 2 3
                do
                    for GRID_SIZE in 10
                    do
                        for NET in "PROGRESSIVE"
                        do
                            sleep 3
                            . test_hyperparams.sh $seed $total_reload $CUDA_VISIBLE $TRAIN_EP $Trainstep $LAYERs $EXPAND $BUFSIZE $BAT_SIZE $LR $n_test $N_CHOOSE $FINAL_EPS $Env $GRID_SIZE $UNSELECT $NET $gpu_config $start_reload $end_reload $TARGET_UPDATE $AGENT &
                            # ((CUDA_VISIBLE=CUDA_VISIBLE+5))
                            # ((gpu_config=gpu_config+0.0001))
                        done
                        # . reload_train2.sh $seed $total_reload $CUDA_VISIBLE $TRAIN_EP $Trainstep $LAYERs $EXPAND $BUFSIZE $BAT_SIZE $LR $n_test &
                    done
                done
            done
        done    
        ((CUDA_VISIBLE=CUDA_VISIBLE+1))
    done
done
