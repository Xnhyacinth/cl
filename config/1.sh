
# set -e

for i in {4..6}; do
    b=`python 1.py && echo "true" || echo "false"`
    # echo $b
    if [[ $b != *true* ]];then
        echo "Error"
        exit 1
    else
        echo "true"
    fi
done