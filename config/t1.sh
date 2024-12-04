
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-11-19 11:47:55
### 

datasets="2_3_lower 2_3_upper"
datasets="llama2_2_3_lower llama2_2_3_upper"
datasets="3500_lower 3500_upper"
datasets="2500_lower 2500_upper 3500_lower 3500_upper"
# for item in $datasets; do
#     echo "Item: $item"
#     bash config/run.sh 2 8,9 t5-large lora 4 constant 1e-4 ${item} all 0 16
# done


datasets="2_3_lower 2_3_upper"
datasets="llama2_2_3_lower llama2_2_3_upper"
datasets="llama2_3500_lower llama2_3500_upper"
# for item in $datasets; do
#     echo "Item: $item"
#     bash config/run.sh 1 1 llama2-7b lora 16 constant 1e-4 ${item} all 0 16
# doneanks="4 2"
# ranks="4"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=32
#     fi
#     if [ "$item" = "4" ];then
#         r2=16
#     fi
#     bash config/run4.sh 2 1,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4
# done
ranks="2"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=32
    fi
    if [ "$item" = "4" ];then
        r2=16
    fi
    bash config/run4.sh 2 1,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4
done
ranks="4 2"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=16
    fi
    if [ "$item" = "4" ];then
        r2=8
    fi
    bash config/run4.sh 2 1,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4
done

# nohup bash config/t1.sh > logs/t1.log 2>&1 &