import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
#test

def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def merge_datasets(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Ensure consistent column names and merge on 'dataset'."""
    df2 = df2.rename(columns={'Name of Time Series': 'dataset'})
    return pd.merge(df1, df2, on='dataset', how='inner')


def compute_ad_entries(
    merged: pd.DataFrame,
    method: str
) -> list[dict]:
    """
    For a given method, compute absolute differences (AD) between predictions
    (unscaled, x2, /2) and ground truth for each dataset.
    """
    entries = []
    for dataset, group in merged.groupby('dataset'):
        preds = group[method]
        truth = group['Median']

        scaled2 = preds * 2
        scaled_half = preds / 2

        # Compute AD arrays
        ad = (truth - preds).abs()
        ad2 = (truth - scaled2).abs()
        ad_half = (truth - scaled_half).abs()

        # For each row, record all variants
        for pred, p2, ph, gt, a, a2, ah in zip(
            preds, scaled2, scaled_half, truth, ad, ad2, ad_half
        ):
            entries.append({
                'dataset': dataset,
                'method': method,
                'predicted': pred,
                'predicted*2': p2,
                'predicted/2': ph,
                'ground_truth': gt,
                'ad': a,
                'ad*2': a2,
                'ad/2': ah,
                'min_ad': min(a, a2, ah)
            })
    return entries


def compute_overall_mae(
    merged: pd.DataFrame,
    methods: list[str]
) -> list[dict]:
    """
    Compute MAE (and scaled variants) over all datasets for each method.
    """
    results = []
    for method in methods:
        preds = merged[method]
        truth = merged['Median']

        mae = mean_absolute_error(truth, preds)
        mae2 = mean_absolute_error(truth, preds * 2)
        mae_half = mean_absolute_error(truth, preds / 2)

        results.append({
            'method': method,
            'Mean': preds.mean(),
            'MAE': mae,
            'MAE*2': mae2,
            'MAE/2': mae_half
        })
    return results


def assign_top_ad(ad_df: pd.DataFrame, mae_df: pd.DataFrame) -> pd.DataFrame:
    """
    Based on which scaling yields lowest MAE per method,
    tag each row in the AD DataFrame with its corresponding top AD value.
    """
    # Map MAE column names to AD column names
    col_map = {'MAE': 'ad', 'MAE*2': 'ad*2', 'MAE/2': 'ad/2'}

    # Determine best scaling per method
    best = (
        mae_df.set_index('method')
        [['MAE', 'MAE*2', 'MAE/2']]
        .idxmin(axis=1)
        .to_dict()
    )

    # Apply mapping to annotate AD dataframe
    def top_ad(row):
        best_col = best.get(row['method'], 'MAE')
        return row[col_map[best_col]]

    ad_df['top_ad'] = ad_df.apply(top_ad, axis=1)
    return ad_df


def process_files(
    file1_path: str,
    file2_path: str,
    ad_out: str,
    mae_out: str
):
    """
    Load, merge, compute AD entries and MAEs, annotate, and save results.
    """
    # Load and merge
    df1 = load_csv(file1_path)
    df2 = load_csv(file2_path)
    merged = merge_datasets(df1, df2)

    methods = ['FFT', 'ACF', 'SuSS', 'MWF', 'Autoperiod', 'RobustPeriod', 'ReWin']

    # Compute detailed AD entries
    ad_entries = []
    for m in methods:
        ad_entries.extend(compute_ad_entries(merged, m))
    ad_df = pd.DataFrame(ad_entries)

    # Compute overall MAE
    mae_results = compute_overall_mae(merged, methods)
    mae_df = pd.DataFrame(mae_results)[['method', 'Mean', 'MAE', 'MAE*2', 'MAE/2']]

    # Annotate with top_ad and save
    ad_df = assign_top_ad(ad_df, mae_df)
    ad_df.to_csv(ad_out, index=False)
    mae_df.to_csv(mae_out, index=False)

    print(f"Processing complete for {file2_path}")
    print(f"AD results saved at: {ad_out}")
    print(f"MAE results saved at: {mae_out}\n")


if __name__ == '__main__':
    ws_csv = "experiments/window_sizes_rewin.csv"
    pairs = [
        ("experiments/tssb_gt/ts_gt.csv", "experiments/results/ad_results.csv", "experiments/results/mae_results.csv"),
        ("experiments/tssb_gt/ts_gt-.csv", "experiments/results/ad_results-.csv", "experiments/results/mae_results-.csv"),
    ]
    for file2, ad_file, mae_file in pairs:
        process_files(ws_csv, file2, ad_file, mae_file)
