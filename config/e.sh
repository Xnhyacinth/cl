

ranks="4 3 2 0 5"
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
        item=2
        r2=4
    fi
    if [ "$item" = "3" ];then
        item=2
        r2=8
    fi
    if [ "$item" = "5" ];then
        item=4
        r2=16
    fi
    # bash config/run.sh 1 0 llama2-7b lora 16 constant 1e-4 0 all 0 8 -1
    # bash config/run5.sh 1 0 llama3.1-8b vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 1 2 0 0 16
    bash config/run4.sh 1 1 llama2-7b vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 1 2 0 1
done

# # nohup bash config/e.sh > logs/llama_e.log 2>&1 &