

bash config/run2.sh 4 2,7,8,9 t5-large lora 16 constant 1e-4 0 all 0 8 -1 0 0 0 0 0 0 0 0 0 0 0 1
# ranks="8 4"
# # ranks="16"
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
#     bash config/run5.sh 1 1 qwen2.5-7b lora 4 constant 1e-4 0 all 0 $item -1 0 0 0 0 0 0 0 0 0 0 0 0
#     # bash config/run.sh 1 0 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 $item $r2 0.5 0 8 1 8
# done
# # nohup bash config/t3.sh > logs/llama_t3.log 2>&1 &