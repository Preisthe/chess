for convstd in 0.05 0.1 0.5; do
for linstd in 0.001 0.005 0.1; do
for batchsize in 36 60 120; do
for lr in 0.0001 0.001 0.01; do
    echo $convstd
    python train.py \
        --convstd $convstd \
        --linstd $linstd \
        --batchsize $batchsize \
        --epochs $epochs \
        --lr $lr
    python chessboard.py
done