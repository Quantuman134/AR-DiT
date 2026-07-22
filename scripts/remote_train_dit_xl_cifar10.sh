
source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate dit
cd /scratch/project/prj-02-visual-ai/hkzhang/AR-DiT

bash scripts/train_cifar10.sh configs/train/cifar10_train.yaml