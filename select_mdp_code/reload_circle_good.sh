#!/bin/bash
total_reload=80
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
SORT=0
gpu_config=0.948
BAD_CIRCE=3
TARGET_UPDATE=2500
start_reload=0
end_reload=$total_reload
for N_CHOOSE in 4
do
    for BAT_SIZE in 128
    do
        for LR in 0.001
        do
            for seed in 0
            do
                for n_test in 200 500
                do
                    for NET in "SHAREEINET" "PROGRESSIVE"
                    do
                        for SORT in 0
                        do
                            sleep 3
                            . test_circle_single.sh $seed $total_reload $CUDA_VISIBLE $TRAIN_EP $Trainstep $LAYERs $EXPAND $BUFSIZE $BAT_SIZE $LR $n_test $N_CHOOSE $FINAL_EPS $NET $SORT $gpu_config $BAD_CIRCE $start_reload $total_reload $TARGET_UPDATE &
                            ((CUDA_VISIBLE=CUDA_VISIBLE+1))
                            # ((gpu_config=gpu_config+0.0001))
                        done
                    done
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