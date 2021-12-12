import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle


def benchmark():
    # square size of the test matrices
    sizes = [int(n * pow(10,i)) for i in range(4) for n in np.arange(1, 10, 0.25)]
    repeats = 300
    data = []

    for s in sizes:
        avg = 0.0
        for _ in range(repeats):
            a = cp.random.rand(s, s, dtype=cp.float32)
            b = cp.random.rand(s, s, dtype=cp.float32)
            begin = time.time()
            _ = a * b
            avg += time.time() - begin
        avg /= repeats
        data.append((s*s, avg))
        print(".", end="", flush=True)
    print()
    
    with open("benchmark.pckl", 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    return data
    

def graph(data):
    x_es = []
    y_es = []
    for d in data:
        x_es.append(d[0])
        y_es.append(d[1])
    
    plt.plot(x_es, y_es)
    plt.show()

if __name__ == "__main__":
    data = None
    try:
        data = pickle.load(open("benchmark.pckl", "rb"))
        print("Loaded data. Not performing benchmark.")
    except Exception:
        print("Benchmark data not found. Running benchmark.")
        data = benchmark()
    
    graph(data)
