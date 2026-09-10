import time
from concurrent.futures import Executor

from executorlib import SingleNodeExecutor

from graphatoms.enterpoint.parallel import ProcessPoolExecutor, SerialExecutor


def test_() -> None:
    assert isinstance(SerialExecutor(), Executor)
    assert isinstance(ProcessPoolExecutor(), Executor)
    assert isinstance(SingleNodeExecutor(), Executor)
    print("AAA")

    def func(i: int) -> int:
        time.sleep(2)
        return 2 * i

    with SingleNodeExecutor(4) as executor:
        lst = []
        for _ in range(40):
            lst.append(executor.submit(func, _))
        print("fdsafds")
        for i, f in enumerate(lst):
            if i > 4:
                f.cancel()
            else:
                print(i, f.result())
if __name__ == "__main__":
    test_()
