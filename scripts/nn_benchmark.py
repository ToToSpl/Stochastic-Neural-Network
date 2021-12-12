import sys
sys.path.append(".")
from gpu.StochNN_GPU import StochNN_GPU
import time
import cupy as cp
import matplotlib.pyplot as plt
import pickle

def test(s, repeats, s_input, s_output):
    avg = 0.0
    vec_x = cp.random.rand(s_input, dtype=cp.float32)
    vec_y = cp.random.rand(s_output, dtype=cp.float32)
    nn = StochNN_GPU(s_input, [s, s], s_output)
    for _ in range(repeats):
        begin = time.time()
        pred = nn.feed_forward(vec_x)
        nn.backpropagation(vec_y, pred)
        avg += time.time() - begin
    avg /= repeats
    return avg


def benchmark():
    # square size of the test matrices
    sizes = [i for i in range(100, 6000, 50)]
    repeats = 150
    s_input = 28*28
    s_output = 10
    data = []
    test(100, repeats, s_input, s_output)

    for s in sizes:
        avg = test(s, repeats, s_input, s_output)
        data.append((s, avg))
        print(".", end="", flush=True)
    print()
    
    with open("benchmark_nn.pckl", 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    return data
    

def graph(data):
    x_es = []
    y_es = []
    for d in data:
        x_es.append(d[0])
        y_es.append(1000 * d[1])

    plt.title("Backpropagation time vs hidden layer size")
    plt.xlabel("Single hidden layer size")
    plt.ylabel("Average time (ms)")
    plt.plot(x_es, y_es)
    plt.show()

if __name__ == "__main__":
    data = None
    try:
        data = pickle.load(open("benchmark_nn.pckl", "rb"))
        print("Loaded data. Not performing benchmark.")
    except Exception:
        print("Benchmark data not found. Running benchmark.")
        data = benchmark()
    
    graph(data)


