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
    targets_train, targets_test = targets[:60000], targets[60000:]
    print("Download complete!")

    weights_hidden = np.random.randn(HIDDEN_SIZE, NUM_INPUTS) \
                     * np.sqrt(2 / NUM_INPUTS) # He initalization
    biases_hidden = np.zeros((HIDDEN_SIZE, 1))

    weights_output = np.random.randn(NUM_CLASSES, HIDDEN_SIZE) \
                     * np.sqrt(1 / HIDDEN_SIZE) # Xavier initialization
    biases_output = np.zeros((NUM_CLASSES, 1))

def train(epochs: int = 15, batch_size: int = 64, learning_rate: float = 0.01) -> None:
    num_batches = 60000 // batch_size + 1
    for epoch in range(epochs):
        sample_count = 0
        epoch_done = False
        for batch in range(num_batches):
            if epoch_done:
                break
            for sample in range(batch_size):
                if sample_count >= 60000:
                    epoch_done = True
                    break
                
                sample_count += 1

    

def main() -> None:
    init()
    train(epochs=1)

if __name__ == "__main__":
    main()