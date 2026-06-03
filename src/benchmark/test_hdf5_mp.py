# """Test AseH5DB multi-process read and single-process write capabilities."""

# import multiprocessing
# import os
# import random
# import statistics
# import time
# from pathlib import Path
# from tempfile import TemporaryDirectory

# import numpy as np
# import pytest
# from ase.collections import g2

# from graphatoms.system import SysGraph
# from graphatoms.system._database.hdf5 import AseH5DB


# def _create_test_graph(name: str, idx: int) -> SysGraph:
#     """Create a test SysGraph for database operations.

#     Ensure the graph passes check_minima validation:
#     - fmax < 0.05 (default threshold)
#     - frequencies > 30 (default fqmin threshold)
#     """
#     atoms = g2[name]
#     return SysGraph.from_ase(
#         atoms,
#         energy=2.5 + idx * 0.1,
#         fmax=0.01 + idx * 0.001,
#         frequencies=np.array([51.0, 52.0, 53.0]) + idx * 2,
#     )


# def _reader_worker(
#     db_path: str,
#     keys: list[str],
#     results: multiprocessing.Queue,
# ) -> None:
#     """Worker function for reading from database in parallel."""
#     try:
#         db = AseH5DB(Path(db_path), append=True)
#         for key in keys:
#             if key in db:
#                 atoms = db[key]
#                 results.put(("read_success", key, atoms.info["energy"]))
#             else:
#                 results.put(("read_missing", key))
#         results.put(("done", os.getpid()))
#     except Exception as e:
#         results.put(("error", os.getpid(), str(e)))


# def _writer_worker(
#     db_path: str,
#     graphs: list[SysGraph],
#     results: multiprocessing.Queue,
# ) -> None:
#     """Worker function for writing to database."""
#     try:
#         db = AseH5DB(Path(db_path), append=True)
#         for g in graphs:
#             success = db.add(g)
#             results.put(("write", g.hash, success))
#         results.put(("write_done", os.getpid()))
#     except Exception as e:
#         results.put(("write_error", os.getpid(), str(e)))


# def _consistency_worker(
#     db_path: str,
#     iterations: int,
#     queue: multiprocessing.Queue,
# ) -> None:
#     """Worker function for testing concurrent read consistency."""
#     pid = os.getpid()
#     try:
#         db = AseH5DB(Path(db_path), append=True)
#         for _ in range(iterations):
#             key = random.choice(list(db.keys()))
#             atoms = db[key]
#             queue.put(("read", pid, key, atoms.info["energy"]))
#             time.sleep(0.001)
#         queue.put(("done", pid))
#     except Exception as e:
#         queue.put(("error", pid, str(e)))


# def _continuous_reader_worker(
#     db_path: str,
#     iterations: int,
#     queue: multiprocessing.Queue,
# ) -> None:
#     """Worker function for continuous reading during writes."""
#     pid = os.getpid()
#     try:
#         for _ in range(iterations):
#             db = AseH5DB(Path(db_path), append=True)
#             keys = list(db.keys())
#             if keys:
#                 key = random.choice(keys)
#                 atoms = db[key]
#                 queue.put(("read", pid, key, len(keys)))
#             del db
#             time.sleep(0.005)
#         queue.put(("reader_done", pid))
#     except Exception as e:
#         queue.put(("reader_error", pid, str(e)))


# def _periodic_writer_worker(
#     db_path: str,
#     num_graphs: int,
#     queue: multiprocessing.Queue,
# ) -> None:
#     """Worker function for periodic writing during reads."""
#     pid = os.getpid()
#     try:
#         valid_names = g2.names[10 : 10 + num_graphs]
#         for i, name in enumerate(valid_names):
#             time.sleep(0.01)
#             g = _create_test_graph(name, i + 30)
#             db = AseH5DB(Path(db_path), append=True)
#             success = db.add(g)
#             queue.put(("write", pid, g.hash, success))
#             del db
#         queue.put(("writer_done", pid))
#     except Exception as e:
#         queue.put(("writer_error", pid, str(e)))


# def _perf_reader_worker(
#     db_path: str,
#     keys: list[str],
#     iterations: int,
#     queue: multiprocessing.Queue,
# ) -> None:
#     """Worker function for performance testing of reads."""
#     pid = os.getpid()
#     try:
#         db = AseH5DB(Path(db_path), append=True)
#         start_time = time.time()
#         total_reads = 0
#         for _ in range(iterations):
#             for key in keys:
#                 atoms = db[key]
#                 total_reads += 1
#         elapsed = time.time() - start_time
#         queue.put(("perf_read_done", pid, elapsed, total_reads))
#     except Exception as e:
#         queue.put(("perf_error", pid, str(e)))


# def _perf_writer_worker(
#     db_path: str,
#     graphs: list[SysGraph],
#     queue: multiprocessing.Queue,
# ) -> None:
#     """Worker function for performance testing of writes."""
#     pid = os.getpid()
#     try:
#         db = AseH5DB(Path(db_path), append=True)
#         start_time = time.time()
#         total_writes = 0
#         for g in graphs:
#             db.add(g)
#             total_writes += 1
#         elapsed = time.time() - start_time
#         queue.put(("perf_write_done", pid, elapsed, total_writes))
#     except Exception as e:
#         queue.put(("perf_error", pid, str(e)))


# class TestAseH5DBMultiprocess:
#     """Test multi-process read and single-process write operations."""

#     @pytest.fixture(scope="class")
#     def test_db_path(self):
#         """Create a temporary database path."""
#         with TemporaryDirectory() as tmpdir:
#             yield Path(tmpdir) / "test_mp.h5"

#     def test_multiple_readers_single_writer(self, test_db_path: Path):
#         """Test multiple processes reading while single process writes."""
#         db = AseH5DB(test_db_path, append=False)
#         initial_graphs = [
#             _create_test_graph(name, i) for i, name in enumerate(g2.names[:10])
#         ]
#         for g in initial_graphs:
#             db.add(g)
#         del db

#         db = AseH5DB(test_db_path, append=True)
#         all_keys = list(db.keys())
#         del db
#         assert len(all_keys) == 10

#         num_readers = 4
#         results_queue = multiprocessing.Queue()
#         reader_processes = []

#         keys_per_reader = [all_keys[i::num_readers] for i in range(num_readers)]

#         for i in range(num_readers):
#             p = multiprocessing.Process(
#                 target=_reader_worker,
#                 args=(str(test_db_path), keys_per_reader[i], results_queue),
#             )
#             reader_processes.append(p)
#             p.start()

#         reader_results = []
#         completed_readers = 0
#         while completed_readers < num_readers:
#             result = results_queue.get(timeout=30)
#             if result[0] == "done":
#                 completed_readers += 1
#             elif result[0] == "error":
#                 pytest.fail(f"Reader process {result[1]} failed: {result[2]}")
#             reader_results.append(result)

#         successful_reads = [r for r in reader_results if r[0] == "read_success"]
#         assert len(successful_reads) == 10, (
#             f"Expected 10 successful reads, got {len(successful_reads)}"
#         )

#         results_queue = multiprocessing.Queue()

#         reader_processes = []
#         for i in range(num_readers):
#             p = multiprocessing.Process(
#                 target=_reader_worker,
#                 args=(str(test_db_path), keys_per_reader[i], results_queue),
#             )
#             reader_processes.append(p)
#             p.start()

#         new_graphs = [
#             _create_test_graph(name, i + 10)
#             for i, name in enumerate(g2.names[10:15])
#         ]
#         writer_process = multiprocessing.Process(
#             target=_writer_worker,
#             args=(str(test_db_path), new_graphs, results_queue),
#         )
#         writer_process.start()

#         completed_readers = 0
#         writer_done = False
#         all_results = []

#         while completed_readers < num_readers or not writer_done:
#             result = results_queue.get(timeout=30)
#             all_results.append(result)
#             if result[0] == "done":
#                 completed_readers += 1
#             elif result[0] == "write_done":
#                 writer_done = True
#             elif result[0] == "error" or result[0] == "write_error":
#                 pytest.fail(f"Process {result[1]} failed: {result[2]}")

#         for p in reader_processes:
#             p.join(timeout=10)
#         writer_process.join(timeout=10)

#         write_results = [r for r in all_results if r[0] == "write"]
#         assert len(write_results) == 5, (
#             f"Expected 5 writes, got {len(write_results)}"
#         )

#         db = AseH5DB(test_db_path, append=True)
#         assert len(db) == 15, f"Expected 15 entries, got {len(db)}"

#         for g in initial_graphs + new_graphs:
#             assert g.hash in db, f"Key {g.hash} not found in database"
#             atoms = db[g.hash]
#             assert abs(atoms.info["energy"] - g.energy) < 0.001, (
#                 "Energy mismatch"
#             )

#     def test_concurrent_read_consistency(self, test_db_path: Path):
#         """Test that concurrent reads return consistent data."""
#         db = AseH5DB(test_db_path, append=False)
#         test_graphs = [
#             _create_test_graph(name, i) for i, name in enumerate(g2.names[:20])
#         ]
#         for g in test_graphs:
#             db.add(g)
#         del db

#         num_processes = 5
#         num_iterations = 10
#         results_queue = multiprocessing.Queue()
#         processes = []

#         for _ in range(num_processes):
#             p = multiprocessing.Process(
#                 target=_consistency_worker,
#                 args=(str(test_db_path), num_iterations, results_queue),
#             )
#             processes.append(p)
#             p.start()

#         results: dict[str, list[float]] = {}
#         completed = 0
#         while completed < num_processes:
#             result = results_queue.get(timeout=30)
#             if result[0] == "done":
#                 completed += 1
#             elif result[0] == "error":
#                 pytest.fail(f"Worker {result[1]} failed: {result[2]}")
#             elif result[0] == "read":
#                 _, _, key, energy = result
#                 if key not in results:
#                     results[key] = []
#                 results[key].append(energy)

#         for p in processes:
#             p.join(timeout=10)

#         for key, energies in results.items():
#             unique_energies = set(energies)
#             assert len(unique_energies) == 1, (
#                 f"Inconsistent energy values for key {key}: "
#                 f"got {len(unique_energies)} unique values out of {len(energies)} reads"
#             )

#     def test_write_during_read_race_condition(self, test_db_path: Path):
#         """Test that writes during reads don't corrupt the database."""
#         db = AseH5DB(test_db_path, append=False)
#         initial_graphs = [
#             _create_test_graph(name, i) for i, name in enumerate(g2.names[:5])
#         ]
#         for g in initial_graphs:
#             db.add(g)
#         del db

#         results_queue = multiprocessing.Queue()
#         processes = []

#         for _ in range(3):
#             p = multiprocessing.Process(
#                 target=_continuous_reader_worker,
#                 args=(str(test_db_path), 20, results_queue),
#             )
#             processes.append(p)
#             p.start()

#         writer_p = multiprocessing.Process(
#             target=_periodic_writer_worker,
#             args=(str(test_db_path), 5, results_queue),
#         )
#         processes.append(writer_p)
#         writer_p.start()

#         reader_done = 0
#         writer_done = False
#         while reader_done < 3 or not writer_done:
#             result = results_queue.get(timeout=30)
#             if result[0] == "reader_done":
#                 reader_done += 1
#             elif result[0] == "writer_done":
#                 writer_done = True
#             elif result[0] in ("reader_error", "writer_error"):
#                 pytest.fail(f"Process {result[1]} failed: {result[2]}")

#         for p in processes:
#             p.join(timeout=10)

#         db = AseH5DB(test_db_path, append=True)
#         assert len(db) == 10, (
#             f"Expected 10 entries (5 initial + 5 written), got {len(db)}"
#         )

#         for g in initial_graphs:
#             assert g.hash in db, f"Initial key {g.hash} missing"
#             atoms = db[g.hash]
#             assert abs(atoms.info["energy"] - g.energy) < 0.001

#     def test_read_performance_scalability(self, test_db_path: Path):
#         """Test read performance scalability with multiple processes."""
#         num_entries = 20
#         db = AseH5DB(test_db_path, append=False)
#         test_graphs = [
#             _create_test_graph(name, i)
#             for i, name in enumerate(g2.names[:num_entries])
#         ]
#         for g in test_graphs:
#             db.add(g)
#         del db

#         db = AseH5DB(test_db_path, append=True)
#         all_keys = list(db.keys())
#         del db

#         print(f"\n{'=' * 60}")
#         print("HDF5 READ PERFORMANCE SCALABILITY TEST")
#         print(f"{'=' * 60}")
#         print(f"Number of entries: {num_entries}")
#         print(f"Number of keys per read: {len(all_keys)}")
#         print()

#         results: dict[int, tuple[float, float]] = {}

#         for num_processes in [1, 2, 4, 8]:
#             iterations = 100 // num_processes
#             if iterations < 1:
#                 iterations = 1

#             results_queue = multiprocessing.Queue()
#             processes = []

#             keys_per_process = [
#                 all_keys[i::num_processes] for i in range(num_processes)
#             ]

#             start_time = time.time()
#             for i in range(num_processes):
#                 p = multiprocessing.Process(
#                     target=_perf_reader_worker,
#                     args=(
#                         str(test_db_path),
#                         keys_per_process[i],
#                         iterations,
#                         results_queue,
#                     ),
#                 )
#                 processes.append(p)
#                 p.start()

#             total_reads = 0
#             elapsed_times = []
#             completed = 0
#             while completed < num_processes:
#                 result = results_queue.get(timeout=30)
#                 if result[0] == "perf_read_done":
#                     _, pid, elapsed, reads = result
#                     total_reads += reads
#                     elapsed_times.append(elapsed)
#                     completed += 1
#                 elif result[0] == "perf_error":
#                     pytest.fail(f"Process {result[1]} failed: {result[2]}")

#             for p in processes:
#                 p.join(timeout=10)

#             total_elapsed = time.time() - start_time
#             throughput = total_reads / total_elapsed

#             results[num_processes] = (total_elapsed, throughput)

#             print(
#                 f"Processes: {num_processes:2d} | "
#                 f"Iterations: {iterations:3d} | "
#                 f"Elapsed: {total_elapsed:.4f}s | "
#                 f"Total reads: {total_reads:5d} | "
#                 f"Throughput: {throughput:.2f} reads/s"
#             )

#         print()
#         print(f"{'=' * 60}")
#         print("PERFORMANCE SCALABILITY SUMMARY")
#         print(f"{'=' * 60}")
#         baseline_time = results[1][0]
#         baseline_throughput = results[1][1]

#         print(f"Baseline (1 process): {baseline_throughput:.2f} reads/s")
#         for num_processes in sorted(results.keys()):
#             elapsed, throughput = results[num_processes]
#             speedup = baseline_time / elapsed
#             efficiency = speedup / num_processes * 100
#             print(
#                 f"Processes: {num_processes:2d} | "
#                 f"Throughput: {throughput:.2f} reads/s | "
#                 f"Speedup: {speedup:.2f}x | "
#                 f"Efficiency: {efficiency:.1f}%"
#             )

#         print()
#         print(f"{'=' * 60}")
#         print("Single Process vs Multi-Process Read Performance Comparison")
#         print(f"{'=' * 60}")
#         print(
#             f"| {'Processes':^8} | {'Throughput':^12} | {'vs Single':^12} "
#             f"| {'Speedup':^8} | {'Efficiency':^8} |"
#         )
#         print(f"|{'-' * 10}|{'-' * 14}|{'-' * 14}|{'-' * 10}|{'-' * 10}|")

#         for num_processes in sorted(results.keys()):
#             elapsed, throughput = results[num_processes]
#             speedup = baseline_time / elapsed
#             efficiency = speedup / num_processes * 100
#             vs_single = (
#                 (throughput - baseline_throughput) / baseline_throughput * 100
#             )
#             vs_symbol = "↑" if vs_single > 0 else "↓" if vs_single < 0 else "="

#             print(
#                 f"| {num_processes:^8} | "
#                 f"{throughput:>10.2f} reads/s | "
#                 f"{vs_symbol} {abs(vs_single):>8.1f}% | "
#                 f"{speedup:>6.2f}x | "
#                 f"{efficiency:>6.1f}% |"
#             )

#         print()
#         print("Comparison Analysis:")
#         print(f"- Single process baseline: {baseline_throughput:.2f} reads/s")
#         max_processes = max(results.keys())
#         max_throughput = max([r[1] for r in results.values()])
#         best_processes = [
#             k for k, v in results.items() if v[1] == max_throughput
#         ][0]
#         print(
#             f"- Maximum throughput: {max_throughput:.2f} reads/s "
#             f"(using {best_processes} processes)"
#         )
#         print(
#             f"- With {max_processes} processes: throughput is "
#             f"{results[max_processes][1] / baseline_throughput * 100:.1f}"
#             f"% of single process"
#         )

#         if max_processes > 1:
#             scaling_efficiency = (
#                 (results[max_processes][1] / baseline_throughput)
#                 / max_processes
#                 * 100
#             )
#             print(f"- Overall scaling efficiency: {scaling_efficiency:.1f}%")
#             if scaling_efficiency < 50:
#                 print(
#                     "  (Note: Multi-process efficiency drop may be due to "
#                     "HDF5 lock contention or I/O bottleneck)"
#                 )

#     def test_write_performance(self, test_db_path: Path):
#         """Test write performance with single process."""
#         print(f"\n{'=' * 60}")
#         print("HDF5 WRITE PERFORMANCE TEST")
#         print(f"{'=' * 60}")

#         num_trials = 3
#         num_entries = 10

#         times = []
#         throughputs = []

#         for trial in range(num_trials):
#             db = AseH5DB(test_db_path, append=False)
#             del db

#             start_idx = trial * num_entries
#             end_idx = start_idx + num_entries
#             test_graphs = [
#                 _create_test_graph(g2.names[i], i)
#                 for i in range(start_idx, min(end_idx, len(g2.names)))
#             ]

#             results_queue = multiprocessing.Queue()
#             p = multiprocessing.Process(
#                 target=_perf_writer_worker,
#                 args=(str(test_db_path), test_graphs, results_queue),
#             )

#             start_time = time.time()
#             p.start()
#             p.join(timeout=30)

#             result = results_queue.get(timeout=5)
#             if result[0] == "perf_write_done":
#                 _, pid, elapsed, total_writes = result
#                 throughput = total_writes / elapsed
#                 times.append(elapsed)
#                 throughputs.append(throughput)
#                 print(
#                     f"Trial {trial + 1}: {elapsed:.4f}s "
#                     f"| {throughput:.2f} writes/s"
#                 )
#             elif result[0] == "perf_error":
#                 pytest.fail(f"Process {result[1]} failed: {result[2]}")

#         print()
#         print(f"{'=' * 60}")
#         print("WRITE PERFORMANCE SUMMARY")
#         print(f"{'=' * 60}")
#         print(f"Number of entries per trial: {num_entries}")
#         print(f"Number of trials: {num_trials}")
#         print(f"Average time: {statistics.mean(times):.4f}s")
#         print(
#             f"Average throughput: {statistics.mean(throughputs):.2f} writes/s"
#         )
#         print(f"Min time: {min(times):.4f}s")
#         print(f"Max time: {max(times):.4f}s")
#         print(f"Std dev: {statistics.stdev(times):.4f}s")

#     def test_mixed_workload_throughput(self, test_db_path: Path):
#         """Test throughput under mixed read/write workload."""
#         db = AseH5DB(test_db_path, append=False)
#         initial_graphs = [
#             _create_test_graph(name, i) for i, name in enumerate(g2.names[:15])
#         ]
#         for g in initial_graphs:
#             db.add(g)
#         del db

#         db = AseH5DB(test_db_path, append=True)
#         read_keys = list(db.keys())
#         del db

#         print(f"\n{'=' * 60}")
#         print("HDF5 MIXED WORKLOAD THROUGHPUT TEST")
#         print(f"{'=' * 60}")
#         print(f"Initial entries: {len(read_keys)}")

#         write_graphs = [
#             _create_test_graph(g2.names[i], i)
#             for i in range(15, min(25, len(g2.names)))
#         ]

#         results_queue = multiprocessing.Queue()
#         processes = []

#         for _ in range(3):
#             p = multiprocessing.Process(
#                 target=_perf_reader_worker,
#                 args=(str(test_db_path), read_keys, 50, results_queue),
#             )
#             processes.append(p)
#             p.start()

#         p = multiprocessing.Process(
#             target=_perf_writer_worker,
#             args=(str(test_db_path), write_graphs, results_queue),
#         )
#         processes.append(p)
#         p.start()

#         start_time = time.time()
#         total_reads = 0
#         total_writes = 0
#         readers_done = 0
#         writer_done = False

#         while readers_done < 3 or not writer_done:
#             result = results_queue.get(timeout=30)
#             if result[0] == "perf_read_done":
#                 _, pid, elapsed, reads = result
#                 total_reads += reads
#                 readers_done += 1
#             elif result[0] == "perf_write_done":
#                 _, pid, elapsed, writes = result
#                 total_writes += writes
#                 writer_done = True
#             elif result[0] == "perf_error":
#                 pytest.fail(f"Process {result[1]} failed: {result[2]}")

#         total_elapsed = time.time() - start_time

#         for p in processes:
#             p.join(timeout=10)

#         print()
#         print(f"{'=' * 60}")
#         print("MIXED WORKLOAD RESULTS")
#         print(f"{'=' * 60}")
#         print(f"Total elapsed time: {total_elapsed:.4f}s")
#         print(f"Total reads: {total_reads}")
#         print(f"Total writes: {total_writes}")
#         print(f"Read throughput: {total_reads / total_elapsed:.2f} reads/s")
#         print(f"Write throughput: {total_writes / total_elapsed:.2f} writes/s")
#         print(
#             f"Combined throughput: "
#             f"{(total_reads + total_writes) / total_elapsed:.2f} ops/s"
#         )


# if __name__ == "__main__":
#     pytest.main([__file__, "-vv", "-s"])
