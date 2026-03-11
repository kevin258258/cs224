#!/bin/bash

# CS224n 作业 3 - 学生提交打包脚本
# 该脚本会将你的实现打包为可提交到 Gradescope 的压缩包

echo "正在创建学生提交压缩包..."

# 提交所需文件
FILES_TO_INCLUDE=(
    "model_solution.py"
    "utils.py"
    "train.py"
)

# 检查所需文件是否存在
missing_files=()
for file in "${FILES_TO_INCLUDE[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "错误：缺少以下必需文件："
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo "请先确认所有必需文件都已存在，再创建提交包。"
    exit 1
fi

# 创建提交压缩包
zip_name="submission.zip"
rm -f "$zip_name"

# 仅添加必需文件
for file in "${FILES_TO_INCLUDE[@]}"; do
    if [ -f "$file" ]; then
        echo "添加文件：$file"
        zip -q "$zip_name" "$file"
    fi
done

echo ""
echo "✓ 已创建提交包：$zip_name"
echo "✓ 可以上传到 Gradescope 了！"
echo ""
echo "包含文件："
unzip -l "$zip_name"
