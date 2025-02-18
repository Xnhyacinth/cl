ranks="3 2 0 5"
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
    bash config/run5.sh 4 1,2,5,6 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 1 4 0 0 0 1
done

# nohup bash config/rs.sh > logs/rs.log 2>&1 &