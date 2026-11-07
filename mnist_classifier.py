import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml

inputs_train: npt.NDArray | None = None
inputs_test: npt.NDArray | None = None
targets_train: npt.NDArray | None = None
targets_test: npt.NDArray | None = None

weights_hidden: npt.NDArray | None = None
biases_hidden: npt.NDArray | None = None

weights_output: npt.NDArray | None = None
biases_output: npt.NDArray | None = None

NUM_INPUTS: int = 784 
HIDDEN_SIZE: int = 128
NUM_CLASSES: int = 10

def init() -> None:
    global inputs_train, inputs_test, targets_train, targets_test
    global weights_hidden, biases_hidden, weights_output, biases_output

    print("Downloading MNIST")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    inputs = mnist.data.astype(np.float32)
    inputs /= 255.0
    targets = mnist.target.astype(np.uint8)
    inputs_train, inputs_test = inputs[:60000], inputs[60000:]
    targets_train, targets_test = targets[:60000], inputs[60000:]
    print("Download complete!")

    weights_hidden = np.random.randn(HIDDEN_SIZE, NUM_INPUTS) \
                     * np.sqrt(2 / NUM_INPUTS) # He initalization
    biases_hidden = np.zeros((HIDDEN_SIZE, 1))

    weights_output = np.random.randn(NUM_CLASSES, HIDDEN_SIZE) \
                     * np.sqrt(1 / HIDDEN_SIZE) # Xavier initialization
    biases_output = np.zeros((NUM_CLASSES, 1))

    print(weights_hidden)
    print(biases_hidden)
    print(weights_output)
    print(biases_output)

def main() -> None:
    init()

if __name__ == "__main__":
    main()