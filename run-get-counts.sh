set -x

total_count=$1
file_name=$2


for ((i=0; i<total_count; i++)); do
    nohup python src/$dataset/get-counts.py $total_count $i $file_name >/dev/null 2>&1 &
    sleep 0.1
done
