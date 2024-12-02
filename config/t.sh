

datasets="2_3_lower 2_3_upper"
datasets="llama2_2_3_lower llama2_2_3_upper"
datasets="llama2_3500_lower llama2_3500_upper"
# for item in $datasets; do
#     echo "Item: $item"
#     bash config/run.sh 1 1 llama2-7b lora 16 constant 1e-4 ${item} all 0 16
# done

# ranks="2"
# for item in $ranks; do
#     echo "Item: $item"
#     if [ "$item" = "2" ];then
#         r2=16
#     fi
#     if [ "$item" = "4" ];then
#         r2=8
#     fi
#     bash config/run0.sh 1 0 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 $item $r2 
# done
ranks="4"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=32
    fi
    if [ "$item" = "4" ];then
        r2=16
    fi
    bash config/run0.sh 1 0 llama2-7b vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0.5
done
ranks="2"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=32
    fi
    if [ "$item" = "4" ];then
        r2=16
    fi
    bash config/run1.sh 1 0 llama2-7b vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0.5
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
    bash config/run1.sh 1 0 llama2-7b vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0.5
done

# nohup bash config/t.sh > logs/llama2_2_3.log 2>&1 &
# nohup bash config/t.sh > logs/llama2_2_3.log 2>&1 &