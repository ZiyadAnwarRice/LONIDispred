
1) module load python
2) srun --job-name=Dispred --nodes=1 --ntasks=1 --cpus-per-task=32 --partition=gpu2 --gres=gpu:1 --time=01:00:00 --account=loni_disorder01 --pty /bin/bash
3) python -m venv .venv (only if environment has not been created)
4) source ~/.venv/bin/activate
5) pip install -r requirements.txt
6) sbatch LONIDispred.batch
