#!/bin/bash
# read arguments when running the script
echo "Arguments full train: $@"
org_args=""
orginal_epoch=0
dataset=""
# filter argument --epoch from the arguments and assign it to epoch variable
for i in "$@"
do
    case $i in
        --epoch=*)
        epoch="${i#*=}"
        orginal_epoch=$epoch
        shift
        ;;
        --dataset=*)
        dataset="${i#*=}"
        ;;
        *)
        org_args="$org_args $i"
        ;;
    esac
done




# first run the pretrain mode
# epoch is set to 30
epoch=30
mkdir ./result
# put epoch argument to the start of the arguments
python FixMatch_2_supervise.py $org_args --epoch=$epoch

mv ./result ./result_pretrain
mkdir ./result
epoch=$orginal_epoch
# put epoch argument to the start of the arguments
python FixMatch_2.py $org_args --load-model "./result_pretrain/best_model.pth" --load-model-ema "./result_pretrain/best_model.pth" --epoch=$epoch
