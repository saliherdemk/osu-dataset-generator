import argparse

from TimingModel import TimingModel

from Dataset import createDataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", required=True)
    parser.add_argument("--batch_size", default=16)
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--load_from", default=None)
    parser.add_argument("--save_to", default=None)
    parser.add_argument("--mode", default="train")
    parser.add_argument("--lr", default=1e-4, type=float)

    args = parser.parse_args()

    dataloader = createDataLoader(args.dataset_folder, int(args.batch_size))
    for data in dataloader:
        print(data["has_hit"].shape)
        print(data["start_offsets"].shape)
        print(data["end_offsets"].shape)

        model = TimingModel()
        print(model(data["audio"], data["difficulty_rating"]).shape)
        break


if __name__ == "__main__":
    main()
