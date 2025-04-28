#!/bin/bash

git add .
git commit -m "update"
git push origin main --force

echo "提交完成! 按任意键继续..."
read -n 1 -s