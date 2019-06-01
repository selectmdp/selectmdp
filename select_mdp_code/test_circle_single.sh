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
Command=1
N_FEATURE=3
Net=${14}
sort=${15}
gpu=${16}
BADCIR=${17}
start_reload=${18}
end_reload=${19}
target_update=${20}
Env=${21}
N_COMMAND=${22}
agent=${23}
RELOAD=$start_reload
while [ $RELOAD -lt $end_reload ];
do
    # echo "hello world $seed"
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE python train.py --TRAIN_STEP $Trainstep --TRAIN_EPISODE $Trainepisode --AGENT dqn --NETWORK SHAREEINET --N_EQUIVARIANT $n_test --LEARNING_RATE $LR --NWRK_EXPAND $EXPAND --BUFFER_SIZE $BUFSIZE --BATCH_SIZE $BAT_SIZE --LAYERS $LAYERs --SEED $seed --RELOAD_EP $RELOAD --TOTAL_RELOAD $total_reload --N_CHOOSE $nchoose --FINAL_EPS $eps  --ENVIRONMENT $Env --N_SUBACT $Command --N_FEATURES $N_FEATURE --NETWORK $Net --SORTED $sort --N_INVARIANT $BADCIR --GPU_OPTION $gpu --UPDATE_TARGET $target_update --N_SUBACT $N_COMMAND --AGENT $agent
    ret=$?
    if [ $ret -ne 0 ]; 
    then
    #Handle failure
    #exit if required
        exit 1
    fi
    ((RELOAD=RELOAD+1))
done
