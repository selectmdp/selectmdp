#!/bin/bash
BUFSIZE=$8
LAYERs=$6
EXPAND=$7
Trainstep=$5
Trainepisode=$4
CUDA_VISIBLE=$3
seed=$1
total_reload=$2
BAT_SIZE=$9
LR=${10}
n_test=${11}
nchoose=${12}
eps=${13}
RELOAD=0
Env=${14}
GRID_SIZE=${15}
UNSELECT=${16}
NET=${17}
gpu_config=${18}
start_reload=${19}
end_reload=${20}
target_update=${21}
agent=${22}
feature=2
RELOAD=$start_reload
while [ $RELOAD -lt $end_reload ];
do
    # echo "hello world $seed"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE python train.py --ENVIRONMENT $Env --TRAIN_STEP  $Trainstep --TRAIN_EPISODE $Trainepisode --AGENT dqn --NETWORK $NET --N_EQUIVARIANT $n_test --N_INVARIANT $UNSELECT --LEARNING_RATE $LR --NWRK_EXPAND $EXPAND --BUFFER_SIZE $BUFSIZE --BATCH_SIZE $BAT_SIZE --LAYERS $LAYERs --SEED $seed --RELOAD_EP $RELOAD --TOTAL_RELOAD $total_reload --N_CHOOSE $nchoose --FINAL_EPS $eps --GPU_OPTION $gpu_config --GRID_SIZE $GRID_SIZE --UPDATE_TARGET $target_update --AGENT $agent --N_FEATURES $feature
    ret=$?
    if [ $ret -ne 0 ]; 
    then
    #Handle failure
    #exit if required
        exit 1
    fi
    ((RELOAD=RELOAD+1))
done
