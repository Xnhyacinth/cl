

ranks="4 2 0 3 5"
bash patch/install.sh
ranks="2"
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
    bash config/run3.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 100 4
done
ranks="3"
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
        item=1
        r2=8
    fi
    bash config/run4.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 100 4
done

ranks="0 5"
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
    bash config/run.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 8 1 100 4
done
# nohup bash config/t1.sh > logs/t5_t1.log 2>&1 &