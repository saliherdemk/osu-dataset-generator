import argparse

from Seq2Seq import Seq2Seq

from Dataset import createDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train")

    args = parser.parse_args()

    dataloader = createDataLoader(args.dataset_folder, int(args.batch_size))

    model = Seq2Seq()

    epochs = 0

    if args.load_from:
        epochs = model.load_checkpoint(args.load_from)

    if args.mode == "train":
        model.train(dataloader, args.save_to, args.num_epochs, epochs)


if __name__ == "__main__":
    main()
