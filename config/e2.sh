

ranks="4"
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
    bash config/run2.sh 4 0,5,6,7 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 10 4 0 0 0 1
done
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
    bash config/run4.sh 4 0,5,6,7 t5-large vida 1 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 10 4 0 0 0 1
done
# ranks="5 0"
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
#         item=2
#         r2=4
#     fi
#     if [ "$item" = "3" ];then
#         item=2
#         r2=8
#     fi
#     if [ "$item" = "5" ];then
#         item=4
#         r2=16
#     fi
#     bash config/run.sh 2 4,5 t5-large vida 2 constant 1e-4 0 all 0 8 -1 $item $r2 0 0 4 1 10
# done
# nohup bash config/e2.sh > logs/e2.log 2>&1 &