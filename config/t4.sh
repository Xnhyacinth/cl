

# ranks="128 256"
# # ranks="32 1"
# for item in $ranks; do
#     echo "Item: $item"
#     # if [ "$item" = "2" ];then
#     #     r2=16
#     # fi
#     # if [ "$item" = "4" ];then
#     #     r2=8
#     # fi
#     # if [ "$item" = "1" ];then
#     #     r2=32
#     # fi
#     # if [ "$item" = "0" ];then
#     #     item=1
#     #     r2=16
#     # fi
#     bash config/run5.sh 1 0 llama2-7b lora 8 constant 1e-4 0 all 0 ${item} -1
#     # bash config/run.sh 1 0 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 $item $r2 0.5 0 8 1 8
# done
# # nohup bash config/t4.sh > logs/t4.log 2>&1 &

bash config/run1.sh 2 8,9 t5-large lora 16 constant 1e-4 0 all 0 8 -1 0 0 0 0 0 0 0 0 0 0 0 1