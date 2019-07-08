import argparse
import os
import pandas as pd
from shutil import copy2


def get_outdir(base_dir, name):
    outdir = os.path.join(base_dir, name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def fileExists(base_dir, filename):
    return os.path.exists(os.path.join(base_dir, filename))


def main():
    parser = argparse.ArgumentParser(description='Cleanup dataset.')
    parser.add_argument('-o', '--outdir', type=str, nargs='?', default='/data/output_cleaned',
        help='Output folder')
    parser.add_argument('-i', '--indir', type=str, nargs='?', default='/data/output',
        help='Input folder where interpolated.csv is located')
    args = parser.parse_args()

    base_outdir = args.outdir
    indir = args.indir

    dataset_indir = indir
    dataset_outdir = base_outdir

    interpolated_csv_file = os.path.join(dataset_indir, 'interpolated.csv')

    df = pd.read_csv(interpolated_csv_file)

    df = df[df.vx > 0.1]

    if not os.path.exists(dataset_outdir):
        os.makedirs(dataset_outdir)


    corruptRows = []

    for index, row in df.iterrows():
        try:
            filename = row.filename
            copy2(os.path.join(dataset_indir, filename), dataset_outdir)
            row.filename = os.path.basename(filename)
        except EnvironmentError:
            corruptRows.append(index)

    df.drop(corruptRows, inplace=True)

    out_csv_path = os.path.join(dataset_outdir, 'interpolated_cleaned.csv')
    df.to_csv(out_csv_path, header=True)


if __name__ == '__main__':
    main()