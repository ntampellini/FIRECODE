import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import plotext as plt

from firecode.calculators._xtb import xtb_opt
from firecode.utils import read_xyz, time_to_string, timing_wrapper

procs_set = {
    'GFN-FF'   : (1, 2, 4, 8),
    'GFN2-XTB' : (4, 8, 12, 16, 32, 48),
    # 'AIMNET2'  : (1, 2, 4, 8, 12),
    }

# from aimnet2_firecode.interface import aimnet2_worker
# import torch

threads_set = {1, 2,
               4, 6, 8,
               12, 16, 32, 48,
               }

def run_concurrent_test(filename):
    '''
    Run geometry opitmizations on filename with various
    processors and threads settings to determine the best 
    configuration to use in productive runs.

    '''

    data = read_xyz(filename)

    cpus = len(os.sched_getaffinity(0))
    # device = 'gpu' if  torch.cuda.is_available() else 'cpu'

    # In case there are more than 48 CPUs
    threads_set.add(cpus)

    print(f'--> Detected {cpus} CPUs (considering hyperthreading)')

    for method in procs_set.keys():
        print(f'\nStarting {method} test...')

        timings = {p:[] for p in procs_set[method]}
        thread_info = {p:[] for p in procs_set[method]}
        best_avg_time = 1E8
        best_settings = None

        for procs in reversed(procs_set[method]): 
            for threads in threads_set:

                # only run test if we are using between 50% and 100% of cpu resources
                if cpus/2 <= procs * threads <= cpus:

                    processes = []
                    cum_exec_time = 0
                    # gpu_rank = i % threads if device == 'gpu' else None

                    with ProcessPoolExecutor(max_workers=threads) as executor:

                        t_start_batch = perf_counter()
                        
                        for i in range(threads):

                            p = executor.submit(
                                    timing_wrapper,
                                    xtb_opt,
                                    # aimnet2_worker,
                                    data.atomcoords[0],
                                    data.atomnos,
                                    method=method,
                                    solvent='THF',
                                    procs=procs,
                                    # maxiter=1000,
                                    conv_thr="tight",
                                    title=f'Candidate_{i}',
                                    # cuda=(device=='gpu'),
                                    # gpu_rank=gpu_rank,
                                )
                            processes.append(p)

                        for i, process in enumerate(as_completed(processes)):
                            _, elapsed = process.result()
                            cum_exec_time += elapsed

                        t_end_batch = perf_counter()
                        elapsed = t_end_batch-t_start_batch
                        avg = elapsed/threads
                        timings[procs].append(avg)
                        thread_info[procs].append(threads)

                        speedup = cum_exec_time/elapsed
                        if avg < best_avg_time:
                            best_avg_time = avg
                            best_settings = (procs, threads)

                        print(f'{method} --> {procs:2} cores/process, {threads:2} threads ({round(speedup, 2):6}x speedup) : {time_to_string(elapsed/threads, digits=3):10} per structure')

        plt.theme("pro")
        plt.title(method)
        # plt.plotsize(75,15)
        x, y =[], []
        for procs in procs_set[method]:
            times = timings.get(procs, None)
            if times is not None:
                for thread, time in zip(thread_info[procs], times):
                    x.append(f'{procs:2}c/{thread}t') # for procs, thread in zip(procs, thread_info[procs])],
                    y.append(round(time, 3))
        
        plt.simple_bar(x, y, width=90, color='red', title=f'{filename}/{method}')

        # plt.xticks(ticks=set(flatten(thread_info.values(), typefunc=int)))
        plt.xlabel("# Threads")
        plt.ylabel("Time per structure (s)")
        plt.show()
        plt.cld()

        print(f'\n--> Suggested settings for {method}: {best_settings[0]} cores, {best_settings[1]} threads')