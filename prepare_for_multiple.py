import argparse
from utils.functions import prepare_for_multiple

def main():
    parser = argparse.ArgumentParser(description="Prepare subset parquet for multiple step attack")
    parser.add_argument('--input_file', type=str, required=True, help='Original full parquet file')
    parser.add_argument('--output_file', type=str, required=True, help='Output parquet with top N samples')
    parser.add_argument('--amount', type=int, default=400, help='Number of top samples to keep')
    parser.add_argument('--drop', action='store_true', help='Drop extra columns if set')
    args = parser.parse_args()

    prepare_for_multiple(
        input_file=args.input_file,
        output_file=args.output_file,
        amount=args.amount,
        drop=args.drop
    )

if __name__ == '__main__':
    main()
