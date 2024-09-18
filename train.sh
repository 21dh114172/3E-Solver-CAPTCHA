# read arguments when running the script
mkdir ./result
echo "Arguments normal train: $@"
# put epoch argument to the start of the arguments
python FixMatch_2.py $@