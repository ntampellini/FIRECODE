import cProfile
from pstats import Stats



def profiled_wrapper(filename, name):

    datafile = f"firecode_{name}_cProfile.dat"
    cProfile.run("Embedder(filename, args.name).run()", datafile)

    with open(f"firecode_{name}_cProfile_output_time.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("time").print_stats()

    with open(f"firecode_{name}_cProfile_output_cumtime.txt", "w") as f:
        p = Stats(datafile, stream=f)
        p.sort_stats("cumtime").print_stats()