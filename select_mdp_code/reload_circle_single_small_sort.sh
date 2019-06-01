#!/bin/bash
total_reload=70
CUDA_VISIBLE=0
TRAIN_EP=20
Trainstep=2500
EXPAND=6
LAYERs=4
BUFSIZE=50000
# N_CHOOSE=0
seed=0
# NET="SHAREEINET"
FINAL_EPS=0.1
NET="VANILLA"
# NET="DIFFEINET"
SORT=1
gpu_config=0.3112
BAD_CIRCE=3
TARGET_UPDATE=1000
start_reload=0
end_reload=$total_reload
# end_reload=$total_reload
#"SHAREEINET" "VANILLA" "DIFFEINET" "PROGRESSIVE"
# Env="circle_env"
Env="circle_env_good"
Agent="CENTRAL_GREEDY"
for N_COMMAND in 5
do
    for BAD_CIRCE in 0
    do
        for N_CHOOSE in 6
        do
            for BAT_SIZE in 32
            do
                for LR in 0.001
                do
                    for seed in 0 1 2 3
                    do
                        for n_test in 50
                        do
                            for NET in "PROGRESSIVE"
                            do
                                for SORT in 0
                                do
                                    sleep 3
                                    . test_circle_single.sh $seed $total_reload $CUDA_VISIBLE $TRAIN_EP $Trainstep $LAYERs $EXPAND $BUFSIZE $BAT_SIZE $LR $n_test $N_CHOOSE $FINAL_EPS $NET $SORT $gpu_config $BAD_CIRCE $start_reload $end_reload $TARGET_UPDATE $Env $N_COMMAND $Agent &
                                    ((CUDA_VISIBLE=CUDA_VISIBLE+1))
                                    # ((gpu_config=gpu_config+0.0001))
                                done
                            done
                            # ((CUDA_VISIBLE=CUDA_VISIBLE+1))

                            # for NET in "SHAREEINET"
                            # do
                            #     . test_circle_single.sh $seed $total_reload $CUDA_VISIBLE $TRAIN_EP $Trainstep $LAYERs $EXPAND $BUFSIZE $BAT_SIZE $LR $n_test $N_CHOOSE $FINAL_EPS $NET $SORT $gpu_config $BAD_CIRCE &
                            # done
                            # ((CUDA_VISIBLE=CUDA_VISIBLE+1))
                        done
                        # ((CUDA_VISIBLE=CUDA_VISIBLE+1))
                    done
                    # done
                done
            done
        done
    done
done
