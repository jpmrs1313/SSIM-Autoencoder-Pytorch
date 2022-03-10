import argparse

class Config:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--train_data_dir", type=str, default=None, help="Train directory folder path")
        self.parser.add_argument("--test_data_dir", type=str, default=None,  help="Test directory folder path")
        self.parser.add_argument("--mask_data_dir", type=str, default=None)
        self.parser.add_argument("--image_size", type=int, default=256, help="Size to reshape the image, the image will be square, height equal to width")
        self.parser.add_argument("--batch_size", type=int, default=64)

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt

  