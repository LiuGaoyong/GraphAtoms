import time

from executorlib import SingleNodeExecutor
from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def f(x):
    time.sleep(1)
    print(f"f({x})")
    return x**2


@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def g(y):
    print(f"g({y})")
    return y + 1


pipeline = Pipeline([f, g])
inputs = {"x": list(range(100))}

t = time.perf_counter()
with SingleNodeExecutor() as exe:
    results = pipeline.map(inputs, executor=exe, auto_subpipeline=True)
    print(results)
    print(results["z"].output.tolist())
print(time.perf_counter() - t)
