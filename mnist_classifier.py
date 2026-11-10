import numpy as np
import numpy.typing as npt
from sklearn.datasets import fetch_openml
import time

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

def init() -> float:
    global inputs_train, inputs_test, targets_train, targets_test
    global weights_hidden, biases_hidden, weights_output, biases_output

    start_time = time.perf_counter()

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

    return time.perf_counter() - start_time

def relu(x: npt.NDArray) -> npt.NDArray:
    return np.maximum(0, x)

def d_relu(x: npt.NDArray) -> npt.NDArray:
    return (x > 0).astype(x.dtype)

def softmax(x: npt.NDArray) -> npt.NDArray:
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def train(epochs: int = 15, batch_size: int = 64, learning_rate: float = 0.01) -> float:
    global weights_hidden, biases_hidden, weights_output, biases_output
    global inputs_train, targets_train

    print("Training...")
    start_time = time.perf_counter()
    
    num_batches = 60000 // batch_size + 1
    for epoch in range(epochs):
        sample_index = 0
        permutation = np.random.permutation(len(inputs_train))
        inputs_train = inputs_train[permutation]
        targets_train = targets_train[permutation]
        epoch_done = False
        for batch in range(num_batches):
            if epoch_done:
                break
            dw_hidden = np.zeros((HIDDEN_SIZE, NUM_INPUTS))
            db_hidden = np.zeros((HIDDEN_SIZE, 1))

            dw_output = np.zeros((NUM_CLASSES, HIDDEN_SIZE))
            db_output = np.zeros((NUM_CLASSES, 1))

            for sample in range(batch_size):
                if sample_index >= 60000:
                    epoch_done = True
                    break

                # forward pass
                input = inputs_train[sample_index]
                input = input.reshape(1, 784)
                z_hidden = input @ weights_hidden.T + biases_hidden.T
                a_hidden = relu(z_hidden)

                z_output = a_hidden @ weights_output.T + biases_output.T
                a_output = softmax(z_output)

                # calculate loss
                target = targets_train[sample_index]
                prediction = a_output[0][target]
                prediction = np.clip(prediction, 1e-15, 1 - 1e-15)
                loss = -1 * np.log10(prediction)

                # backpropagation
                target_one_hot = np.zeros((1, 10))
                target_one_hot[0][target] = 1
                d_error = a_output - target_one_hot

                dw_output += d_error.T @ a_hidden
                db_output += d_error.T

                dw_hidden += (d_error @ weights_output * d_relu(a_hidden)).T @ input
                db_hidden += (d_error @ weights_output * d_relu(a_hidden)).T
                
                sample_index += 1
            # gradient descent
            weights_hidden -= learning_rate * dw_hidden
            biases_hidden -= learning_rate * db_hidden
            weights_output -= learning_rate * dw_output
            biases_output -= learning_rate * db_output

    print("Training Complete!")
    return time.perf_counter() - start_time

def test_accuracy() -> float:
    # forward pass
    z_hidden = inputs_test @ weights_hidden.T + biases_hidden.T
    a_hidden = relu(z_hidden)
    z_output = a_hidden @ weights_output.T + biases_output.T
    a_output = np.zeros((10_000, 10))
    for i, row in enumerate(z_output):
        a_output[i] = softmax(row)

    predictions = np.argmax(a_output, axis=1)
    num_correct = np.sum((predictions == targets_test).astype(int))

    return num_correct / 10_000

def main() -> None:
    print(f"Init Time: {init():.2f}s")
    print(f"Train Time: {train():.2f}s")
    print(f"Accuracy: {test_accuracy():.2%}")

if __name__ == "__main__":
    main()