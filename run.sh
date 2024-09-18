#!/bin/bash
# read arguments when running the script
python_file="pretrain.py"
train_mode=""
hf_token=""
dataset_link=""
dataset=""
dataset_name=""
vocab=""
delimiter=""
label=""
# read arguments when running the script
#echo "Arguments: $*"
org_args=""

# Function to remove leading and trailing quotes
remove_quotes() {
    echo "$1" | sed 's/^"//;s/"$//'
}

remove_equal_if_quotes() {
    if echo "$1" | grep -q '"'; then
        echo "$1" | sed 's/\(.*\)=\(.*"\)/\1 \2/'
    else
        echo "$1"
    fi
}

duplicate="f"
echo $SHELL
echo "Original args $@"
string="$@"
#tokens=(${string//\./ })
#echo "$tokens"
IFS='|'
# Iterate over the tokens
#for i in "${tokens[@]}"
for i in $string
do
    case $i in
        --mode=*)
        mode=$(remove_quotes "${i#*=}")
        train_mode=$mode
        shift
        duplicate="f"
        ;;

        --hf_token=*)
        hf_token=$(remove_quotes "${i#*=}")
        shift
        duplicate="f"
        ;;

        --dataset-link=*)
        dataset_link=$(remove_quotes "${i#*=}")
        shift
        duplicate="f"
        ;;

        --dataset=*)
        dataset=$(remove_quotes "${i#*=}")
        shift
        duplicate="f"
        ;;

        --vocab=*)
        echo $i
        vocab="$(remove_quotes "${i#*=}")"
        ;;
        
        --delimiter-label=*)
        delimiter=$(remove_quotes "${i#*=}")
        shift
        ;;
        --label=*)
        label=$(remove_quotes "${i#*=}")
        shift
        ;;

        # --dataset)
        # dataset="$2"
        # shift 2
        # ;;
        *)
        org_args="$org_args $(remove_equal_if_quotes $i)"
        ;;
    esac
done

IFS=' '

# download the dataset if the dataset-link is not empty
if [ -n "$dataset_link" ]; then
    echo $dataset_link
    wget -O "./dataset.zip" $dataset_link
    # unzip into the dataset folder
    unzip -o -qq "./dataset.zip" -d "./dataset"
    # remove the zip file
    rm "./dataset.zip"
    # get the dataset name from folder in the dataset
    for i in $(ls -d ./dataset/*/); do
        dataset_name=$(basename "$i")
    done
    echo "Dataset name: $dataset_name"
    
fi

echo "$vocab"

echo $args_train
# if mode == "pretrain" then the script will run the pretrain mode with sh full_train.sh "$@"

if [ "$train_mode" = "pretrain" ]; then
    python_file="pretrain.py"
    #args_train="--delimiter-label "${delimiter@Q}" --label $label --vocab "${vocab@Q}" --dataset $dataset_name"$org_args"" ; bash full_train.sh $args_train
    bash full_train.sh --label $label --dataset $dataset_name $org_args
# else if mode == "train" then the script will run the train mode with sh train.sh "$@"
elif [ "$train_mode" = "train" ]; then
    python_file="train.py"
    bash train.sh $args_train
else
    echo "Invalid mode, $train_mode"
    exit 1
fi

# if hf_token is not empty then the script will login to huggingface with the token
if [ -n "$hf_token" ]; then
    huggingface-cli login --token $hf_token
    #7z a -r ./$dataset_name_result.zip ./result > nul
    zip -q -r ./$dataset_name_result.zip ./result > nul
    huggingface-cli upload congminh2456/sulre_pac ./$dataset_name_result.zip .
fi
