import sys
import os
import pandas as pd
import numpy as np
import daproli as dp

sys.path.insert(0, "../")
from data_loader import load_tssb_dataset
from rewin      import rewin

def extend_csv_with_rewin(exp_path, input_csv, n_jobs=1, verbose=0):


    # load csv
    df_ws = pd.read_csv(input_csv, index_col=0)

    # load TSSB
    df_tssb = load_tssb_dataset()
    

    # compute ReWin window sizes
    window_sizes = dp.map(
        rewin,
        df_tssb.time_series,
        expand_args=False,
        ret_type=np.array,
        n_jobs=n_jobs,
        verbose=verbose
    )

    # save window sizes
    df_ws["ReWin"] = window_sizes

    # save new csv
    df_ws.to_csv(input_csv)



if __name__ == "__main__":
    exp_path  = "experiments/"
    input_csv = os.path.join(exp_path, "window_sizes_rewin.csv")
   
   # n_jobs = -1 for parallel
    n_jobs, verbose = 1, 0

    extend_csv_with_rewin(exp_path, input_csv, n_jobs, verbose)
