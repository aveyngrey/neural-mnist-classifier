import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml

# uses np zero array of shape (0, 0) as a placeholder
inputs_train: npt.NDArray = np.zeros((0, 0))
inputs_test: npt.NDArray = np.zeros((0, 0))
targets_train: npt.NDArray = np.zeros((0, 0))
targets_test: npt.NDArray = np.zeros((0, 0))

weights_hidden: npt.NDArray = np.zeros((0, 0))
biases_hidden: npt.NDArray = np.zeros((0, 0))

weights_output: npt.NDArray = np.zeros((0, 0))
biases_output: npt.NDArray = np.zeros((0, 0))

NUM_INPUTS: int = 784 
HIDDEN_SIZE: int = 128
NUM_CLASSES: int = 10

def init() -> None:
    global inputs_train, inputs_test, targets_train, targets_test
    global weights_hidden, biases_hidden, weights_output, biases_output

    print("Downloading MNIST...")
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

def relu(x: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, x)

def softmax(x: npt.NDArray) -> npt.NDArray:
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def train(epochs: int = 15, batch_size: int = 64, learning_rate: float = 0.01) -> None:
    num_batches = 60000 // batch_size + 1
    for epoch in range(1):
        sample_index = 0
        epoch_done = False
        for batch in range(1):
            if epoch_done:
                break
            for sample in range(1):
                if sample_index >= 60000:
                    epoch_done = True
                    break

                input = inputs_train[sample_index]
                z_hidden = input @ weights_hidden.T + biases_hidden.T
                a_hidden = relu(z_hidden)

                z_output = a_hidden @ weights_output.T + biases_output.T
                a_output = softmax(z_output)
                
                sample_index += 1

    

def main() -> None:
    init()
    train(epochs=1)

if __name__ == "__main__":
    main()