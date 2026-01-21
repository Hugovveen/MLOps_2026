import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument(
    "--input_csv",
    type=Path,
    nargs="+",
    required=True,
    help="One or more CSV files to plot together",)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_data(file_path: Path) -> pd.DataFrame:
   return pd.read_csv(file_path)


def setup_style():
    # TODO: Set seaborn theme
    pass


def plot_metrics(csv_paths, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and F1.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        x_col = df.columns[0]
        y_col = df.columns[1]

        # extract seed from path (q4_seed_X)
        seed = csv_path.parent.name.replace("q4_seed_", "")
        ax.plot(df[x_col], df[y_col], label=f"seed {seed}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Global Gradient Norm over Training Steps")
    ax.legend()
    ax.grid(True)

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        out_file = output_path / "grad_norms_3seeds.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_file}")
    else:
        plt.show()

    plt.close()

def main():
    args = parse_args()
    setup_style()
    plot_metrics(args.input_csv, args.output_dir)



if __name__ == "__main__":
    main()
