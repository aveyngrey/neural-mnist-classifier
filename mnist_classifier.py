import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml

inputs_train: npt.NDArray | None = None
inputs_test: npt.NDArray | None = None
targets_train: npt.NDArray | None = None
targets_test: npt.NDArray | None = None



NUM_INPUTS: int = 784 
HIDDEN_LAYER_SIZE: int = 128
NUM_CLASSES: int = 10

def init() -> None:
    global inputs_train, inputs_test, targets_train, targets_test

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    inputs = mnist.data.astype(np.float32)
    inputs /= 255.0
    targets = mnist.targets.astype(np.uint8)
    inputs_train, inputs_test = inputs[:60000], inputs[60000:]
    targets_train, targets_test = targets[:60000], inputs[60000:]



    

def main() -> None:
    init()

if __name__ == "__main__":
    main()