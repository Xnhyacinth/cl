
###
 # Copyright (c) 2024 by Huanxuan Liao, huanxuanliao@gmail.com, All Rights Reserved. 
 # @Author: Xnhyacinth, Xnhyacinth@qq.com
 # @Date: 2024-11-16 07:36:01
### 

bash config/run2.sh 1 1 llama2-7b vida 1 constant 1e-4 0 all 50 8 -1 2 16 0 0 4

# nohup bash config/run.sh 2 1,2 t5-large vida 2 constant 1e-4 0 all 0 8 -1 4 8 > logs/dd.log 2>&1 &
# nohup bash config/run.sh 2 3,4 t5-large vida 2 constant 1e-4 0 all 0 8 -1 2 16 > logs/ee.log 2>&1 &
# nohup bash config/run.sh 2 5,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 4 16 > logs/d.log 2>&1 &
# nohup bash config/run.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 0 8 -1 2 32 > logs/e.log 2>&1 &

# nohup bash config/run.sh 2 1,2 t5-large vida 2 constant 1e-4 0 all 0 8 2 4 16 > logs/dd.log 2>&1 &
# nohup bash config/run.sh 2 3,4 t5-large vida 2 constant 1e-4 0 all 0 8 2 2 32 > logs/ee.log 2>&1 &

# nohup bash config/run1.sh 2 5,7 t5-large vida 2 constant 1e-4 0 all 0 8 -1 4 16 > logs/d.log 2>&1 &
# nohup bash config/run1.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 0 8 -1 2 32 > logs/e.log 2>&1 &
# nohup bash config/run.sh 2 5,7 t5-large vida 2 constant 1e-4 0 all 3500 8 -1 4 16 > logs/a.log 2>&1 &
# nohup bash config/run.sh 2 8,9 t5-large vida 2 constant 1e-4 0 all 4000 8 -1 4 16 > logs/b.log 2>&1 &
# nohup bash config/run.sh 2 0,6 t5-large vida 2 constant 1e-4 0 all 3000 8 -1 4 16 > logs/c.log 2>&1 &
# bash config/run0.sh 1 1 t5-large full 2 constant 1e-4 0 eval 3000 8 -1 4 16
# bash config/run0.sh 1 2 t5-large vida 2 constant 1e-4 0 eval 3000 8 -1 4 16
# bash config/run0.sh 1 6 t5-large vida 2 constant 1e-4 0 eval 3000 8 -1 4 16

# nohup bash config/run0.sh 1 1 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 4 16 > logs/g.log 2>&1 &
# nohup bash config/run0.sh 1 0 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 2 32 > logs/f.log 2>&1 &
# nohup bash config/run0.sh 1 0 llama2-7b vida 4 constant 1e-4 0 all 0 8 -1 2 32 > logs/f.log 2>&1 &


# nohup bash config/run1.sh 2 0,6 t5-large lora 4 constant 1e-4 0 all 0 4 > logs/h.log 2>&1 &
# nohup bash config/run.sh 2 6,7 t5-large lora 4 constant 1e-4 0 all 4000 16 > logs/e.log 2>&1 &

# nohup bash config/run.sh 2 2,3 t5-large lora 4 constant 1e-4 0 all 2500 16 > logs/a.log 2>&1 &
# # nohup bash config/run.sh 2 8,9 t5-large lora 4 constant 1e-4 0 all 3000 16 > logs/c.log 2>&1 &
# nohup bash config/run.sh 2 4,5 t5-large lora 4 constant 1e-4 0 all 2500 1 > logs/b.log 2>&1 &
# nohup bash config/run.sh 2 6,7 t5-large lora 4 constant 1e-4 0 all 3000 1 > logs/d.log 2>&1 &
# nohup bash config/run.sh 2 1,2 t5-large lora 4 > logs/d.log 2>&1 &

# nohup bash config/run.sh 2 3,4 t5-large lora 4 cosine > logs/e.log 2>&1 &

# nohup bash config/run.sh 2 5,7 t5-large lora 4 constant 3e-5 > logs/f.log 2>&1 &

# nohup bash config/run.sh 1 1 llama2-7b lora 16 > logs/a.log 2>&1 &

# nohup bash config/run.sh 1 0 llama2-7b lora 16 cosine > logs/c.log 2>&1 &

# nohup bash config/run.sh 1 1 llama2-7b lora 16 constant 1e-4 0 all 3500 > logs/a.log 2>&1 &

# nohup bash config/run.sh 1 0 llama2-7b lora 16 constant 1e-4 0 all 4000 > logs/b.log 2>&1 &

# nohup bash config/run.sh 2 8,9 tinyllama lora > logs/b.log 2>&1 &
# sudo apt install alipay-linkc-zeta-0.14.0 -b current
# ps -ef |grep config/t1.sh|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep adaprompt|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep scale|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep restore|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep 4,5|grep -v grep |cut -c 9-14|xargs kill -9
# ps -ef |grep restore0.5|grep -v grep |cut -c 9-16|xargs kill -9
# ps -ef |grep 0.5|grep -v grep |cut -c 9-16|xargs kill -9
# ps -ef |grep adaprompt|grep -v grep |cut -c 9-16|xargs kill -9
# datasets="2_3_lower 2_3_upper 4_5_lower 4_5_upper"
# datasets="2_3_lower 2_3_upper"
# for item in $datasets; do
#     echo "Item: $item"
#     bash config/run.sh 2 8,9 t5-large lora 4 constant 1e-4 ${item}
# done
# bash config/e.sh 1 1 t5-large lora 4 constant 1e-4 0 eval
# bash config/e.sh 1 1 t5-large lora 4 constant 1e-4 0 all

# # zeta co https://liaohuanxuan.lhx:6eFp5A9sWrNBxK5LqNDF38g6A@zeta.alipay.com/zeta/cl
# zeta config core.remote https://liaohuanxuan.lhx:6eFp5A9sWrNBxK5LqNDF38g6A@zeta.alipay.com/zeta/cl