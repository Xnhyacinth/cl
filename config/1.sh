
# # set -e

# for i in {4..6}; do
#     b=`python 1.py && echo "true" || echo "false"`
#     # echo $b
#     if [[ $b != *true* ]];then
#         echo "Error"
#         exit 1
#     else
#         echo "true"
#     fi
# done

orders="mnli,cb,wic,copa,qqp,boolqa,rte,imdb,yelp,amazon,sst-2,dbpedia,agnews,multirc,yahoo"
last_element=$(echo $orders | awk -F ',' '{print $NF}')
echo $last_element