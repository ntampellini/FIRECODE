if __name__ == '__main__':
    # ## Let's use malloc to see if we have a memory leak

    # %%

    from firecode.embedder import Embedder

    embedder = Embedder(r'C:\Users\Nik\Desktop\debug\malloc\input', stamp='debug')
    # embedder = Embedder(r'/mnt/c/Users/Nik/Desktop/debug/malloc/input', stamp='debug')
    embedder.objects[0].atomcoords.shape

    # %%
    # does not seem to leak in the conformational search, maybe it does in the augmentation part
    from firecode.embedder import RunEmbedding
    import numpy as np
    mol = embedder.objects[0]

    # embedder.options.ff_calc = 'XTB'
    # embedder.options.ff_level = 'GFN-FF'

    re = RunEmbedding(embedder)
    re.structures = np.array(embedder.objects[0].atomcoords)
    re.atomnos = mol.atomnos
    re.constrained_indices = np.array([[] for _ in re.structures])
    re.energies = np.array([0 ,0])

    re.structures.shape

    # %%

    # re.csearch_augmentation(text='warmup', max_structs=100)

    # # %%
    # tracemalloc.start()
    # s = tracemalloc.take_snapshot()

    # ############################################################

    # try:
    #     # re.csearch_augmentation(text='tracemalloc', max_structs=100)
    #     re.csearch_augmentation_routine()
    # except KeyboardInterrupt:
    #     pass

    # ############################################################

    # lines = []
    # top_stats = tracemalloc.take_snapshot().compare_to(s, 'lineno')
    # for stat in top_stats[:5]:
    #     lines.append(str(stat))
    # print("\n".join(lines))


    # %%
    # from memory_profiler import profile

    # @profile
    def wrapper(*args, **kwargs):
        return re.csearch_augmentation_routine()
        
    try:
        # re.csearch_augmentation(text='tracemalloc', max_structs=100)
        wrapper()
    except KeyboardInterrupt:
        pass