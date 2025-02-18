ranks="4 3 2 0 5"
ranks="4 2 0"
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
    bash config/run5.sh 4 3,4,8,9 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 1 4 0 0 0 0
done

# nohup bash config/t2.sh > logs/t2.log 2>&1 &