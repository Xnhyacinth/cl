
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-11-15 17:08:26
### 
datasets="4_5_lower 4_5_upper"
datasets="llama2_4_5_lower llama2_4_5_upper"
datasets="llama2_4000_lower llama2_4000_upper"
# ranks="16 32 4 2"
# for item in $ranks; do
#     echo "Item: $item"
#     bash config/run1.sh 1 1 llama2-7b lora 16 constant 1e-4 0 all 0 $item
# done
# 
# ranks="4 2"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=32
# #     fi
# #     if [ "$item" = "4" ];then
# #         r2=16
# #     fi
# #     bash config/run1.sh 2 4,5 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0.5
# # done
# ranks="2 4 1 0"
# ranks="0"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=16
#     fi
#     if [ "$item" = "4" ];then
#         r2=8
#     fi
#     if [ "$item" = "1" ];then
#         r2=32
#     fi
#     if [ "$item" = "0" ];then
#         item=1
#         r2=16
#     fi
#     bash config/run0.sh 2 1,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1
# done


ranks="4 2 0 3 5"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=16
    fi
    if [ "$item" = "4" ];then
        r2=8
    fi
    if [ "$item" = "1" ];then
        r2=32
    fi
    if [ "$item" = "0" ];then
        item=1
        r2=16
    fi
    if [ "$item" = "3" ];then
        item=2
        r2=8
    fi
    if [ "$item" = "5" ];then
        item=1
        r2=8
    fi
    bash config/run.sh 2 1,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1
done
# ranks="2"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=16
#     fi
#     if [ "$item" = "4" ];then
#         r2=8
#     fi
#     bash config/run1.sh 2 4,5 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0.5
# done

# nohup bash config/rs.sh > logs/rs.log 2>&1 &