

ranks="4 2 1"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=32
    fi
    if [ "$item" = "4" ];then
        r2=16
    fi
    if [ "$item" = "1" ];then
        r2=32
    fi
    bash config/run4.sh 2 0,6 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4
done
ranks="4 2 1"
for item in $ranks; do
    echo "Item: $item"
    if [ "$item" = "2" ];then
        r2=16
    fi
    if [ "$item" = "4" ];then
        r2=8
    fi
    if [ "$item" = "1" ];then
        r2=16
    fi
    bash config/run4.sh 2 0,6 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4
done