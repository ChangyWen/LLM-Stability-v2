set -x

dataset=$1
total_count=$2


for ((i=0; i<total_count; i++)); do
    nohup python src/$dataset/non-mcq-v2.py $total_count $i >/dev/null 2>&1 &
    sleep 0.1
done
