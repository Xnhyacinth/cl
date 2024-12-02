
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-11-19 11:47:55
### 

datasets="2_3_lower 2_3_upper"
datasets="llama2_2_3_lower llama2_2_3_upper"
datasets="3500_lower 3500_upper"
datasets="3000_lower 3000_upper 4000_lower 4000_upper"
# for item in $datasets; do
#     echo "Item: $item"
#     bash config/run.sh 2 2,3 t5-large lora 4 constant 1e-4 ${item} all 0 16
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
    bash config/run.sh 1 1 llama2-7b vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0.33
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
    bash config/run1.sh 1 1 llama2-7b vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0.33
done

# ranks="4"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         item=17
#         r2=17
#     fi
#     if [ "$item" = "4" ];then
#         item=10
#         r2=10
#     fi
#     bash config/run1.sh 2 2,3 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2
# done
# ranks="4"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=9
#         item=9
#     fi
#     if [ "$item" = "4" ];then
#         r2=6
#         item=6
#     fi
#     bash config/run1.sh 2 2,3 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2
# done

# nohup bash config/t3.sh > logs/t5_2_30.log 2>&1 &