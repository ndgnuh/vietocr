from argparse import ArgumentParser, Namespace
from vietocr.training import Trainer


def parse_args() -> Namespace:
    p = ArgumentParser()

    # Model
    p.add_argument("--lang", required=True)
    p.add_argument("--vocab-type", default="ctc")
    p.add_argument("--image-height", type=int, default=32)
    p.add_argument("--image-min-width", type=int, default=32)
    p.add_argument("--image-max-width", type=int, default=512)


    # Training
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-steps", type=int, default=100_000)
    p.add_argument("--validate-every", type=int, default=2_000)
    p.add_argument("--batch-size", "-b", type=int, default=1)
    p.add_argument("--num-workers", "-p", type=int, default=0)
    p.add_argument("--shuffle", action="store_true", default=False, help="Shuffle data")

    # Data
    p.add_argument("--train-data")
    p.add_argument("--val-data")
    p.add_argument("--test-data")
    return p.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(**args.__dict__)
    trainer.fit()


if __name__ == "__main__":
    main()
