XDATA="[-2,0.5]"
YDATA="[3,-1.0]"
LR=1e-4
HL=2
# python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=1e-2 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=5e-2 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=1e-1 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=5e-1 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=1e0 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=5e0 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=1e1 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
python train.py -F runs with hidden_layers=$HL xdata=$XDATA ydata=$YDATA alpha=5e1 lr=$LR iters=$2 save_every=20 width=$1 linearize=False
