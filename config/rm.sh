#!/bin/bash

# 检查输入参数
# if [ "$#" -lt 2 ]; then
#     echo "Usage: $0 <directory> <suffix>"
#     exit 1
# fi

# # 获取目录和尾缀
# TARGET_DIR=$1
# SUFFIX=$2

# # 检查目录是否存在
# if [ ! -d "$TARGET_DIR" ]; then
#     echo "Error: Directory $TARGET_DIR does not exist."
#     exit 1
# fi

# # 查找并删除匹配的文件或文件夹
# echo "Searching for files and directories ending with '$SUFFIX' in $TARGET_DIR..."
# find "$TARGET_DIR" -name "*$SUFFIX*" -exec rm -rf {} \;

# # 完成
# echo "Deletion completed for files and directories with suffix '$SUFFIX'."

#!/bin/bash

# 检查参数数量
# if [ "$#" -ne 2 ]; then
#     echo "Usage: $0 <target_directory> <search_string> <parent_string>"
#     exit 1
# fi

SEARCH_STRING=${2:-"checkpoint"}
PARENT_STRING=${3:-"0"}
TARGET_DIRECTORY=${1:-"saves/"}

# 查找目标目录下包含特定字符串的文件或文件夹
find "$TARGET_DIRECTORY" -name "*$SEARCH_STRING*" | while read -r file; do
    # 获取文件或文件夹的上级目录
    parent_dir=$(dirname "$file")
    
    # 检查上级目录是否包含额外的字符串
    if [[ $PARENT_STRING == "0" ]]; then
        echo "Removing: $file"
        rm -rf "$file"  # 删除文件或文件夹
    fi
    if [[ $PARENT_STRING != "0" ]]; then
        if [[ $parent_dir != *"$PARENT_STRING"* ]]; then
            echo "Removing: $file"
            rm -rf "$file"  # 删除文件或文件夹
        fi
    fi
done

# bash config/rm.sh saves0/ checkpoint
# bash config/rm.sh saves0/ safetensors 4-