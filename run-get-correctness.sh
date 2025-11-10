set -x

dataset=$1
total_count=$2
file_name=$3


for ((i=0; i<total_count; i++)); do
    nohup python src/$dataset/get-correctness.py $total_count $i $file_name >/dev/null 2>&1 &
    sleep 0.1
done
