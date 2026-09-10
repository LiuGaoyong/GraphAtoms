from concurrent.futures import Executor

from executorlib import SingleNodeExecutor


def test_() -> None:
    # assert isinstance(SerialExecutor(), Executor)
    # assert isinstance(ProcessPoolExecutor(), Executor)
    assert isinstance(SingleNodeExecutor(), Executor)
    print("AAA")


if __name__ == "__main__":
    test_()
