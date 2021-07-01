from pathlib import Path
from nmrespy.load import bruker as bload
from nmrespy.load import load_bruker

FIDPATHS = [Path(f'data/{i}').resolve() for i in range(1, 3)]
PDATAPATHS = [Path(f'data/{i}/pdata/1').resolve() for i in range(1, 3)]
ALLPATHS = FIDPATHS + PDATAPATHS

def test_determine_bruker_data_type():
    for i, path in enumerate(ALLPATHS, start=1):
        info = bload.determine_bruker_data_type(path)
        print(info)
        assert info['dim'] == _funky_mod(i, 2)
        assert info['dtype'] == 'fid' if i < 3 else 'pdata'
        assert list(info['param'].keys()) == (
            [f'acqu{j}s' for j in ['', '2', '3'][:i]]
            if i < 3
            else [f'{t}{j}s' for t in ['acqu', 'proc']
                             for j in ['', '2', '3'][:i - 2]]
        )
        if i == 1:
            assert list(info['bin'].values())[0].name == 'fid'
        elif i == 2:
            assert list(info['bin'].values())[0].name == 'ser'
        else:
            assert list(info['bin'].values())[0].name == \
                f"{info['dim']}{info['dim'] * 'r'}"


def test_load_bruker():
    for path in ALLPATHS:
        info = load_bruker(path)
        print(info)


def _funky_mod(x, r):
    """Same as normal % operator, but if the result is 0, ``r`` is returned
    instead. i.e. 6 % 3 -> 3 rather than the usual 0."""
    m = x % r
    return m if m != 0 else r


# # Plots a 3-dimensional wireframe of the data. Used for testing
# # purposes. Should see a plane of 0's right at the end of the
# # direct dimension
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y = tuple(np.arange(s) for s in data.shape)
# xx, yy = tuple(arr.T for arr in np.meshgrid(x, y))
# ax.plot_wireframe(xx, yy, data)

# # Continuation of testing with plot
# x, y = tuple(np.arange(s) for s in data.shape)
# xx, yy = tuple(arr.T for arr in np.meshgrid(x, y))
# ax.plot_wireframe(xx, yy, data, color='k')
# print('bye')
# plt.show()
