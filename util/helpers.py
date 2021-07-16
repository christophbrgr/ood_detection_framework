import numpy as np

def softmax(outputs, temper=1):
    # Calculating the confidence of the output
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs/temper) / np.sum(np.exp(nnOutputs/temper), axis=1, keepdims=True)
    return nnOutputs
