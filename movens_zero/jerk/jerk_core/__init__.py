
import os

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import torch
from threadpoolctl import threadpool_limits

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
THREAD_LIMIT = threadpool_limits(limits=1)
