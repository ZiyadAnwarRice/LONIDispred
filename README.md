## Create environment
1) module load python (Ensure python version 3.11.5 using python --version)
2) srun --job-name=Dispred --nodes=1 --ntasks=1 --cpus-per-task=32 --partition=gpu2 --gres=gpu:1 --time=01:00:00 --account=loni_disorder01 --pty /bin/bash
3) python -m venv .venv
4) source ~/.venv/bin/activate
5) pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
6) pip install -r requirements.txt


## Run Job
1) source ~/.venv/bin/activate
1) cd /ddnB/work/user/Research/LORADispred
2) sbatch LONIDispred.batch
