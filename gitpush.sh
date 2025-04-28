#!/bin/bash

git add .
git commit -m "脚本上传"
git push origin main --force

echo "提交完成! 按任意键继续..."
read -n 1 -s