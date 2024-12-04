

ranks="2 4 1 0"
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
    bash config/run1.sh 2 0,4 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 1e7
done


# nohup bash config/e2.sh > logs/e2.log 2>&1 &