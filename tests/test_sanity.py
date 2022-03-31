# test_sanity.py
# Simon Hulse
# simon.hulse@chem.ox.ac.uk
# Last Edited: Wed 30 Mar 2022 14:26:05 BST

import pytest
from nmrespy import ExpInfo
from nmrespy._sanity import sanity_check, funcs
import numpy as np


# def test_sanity_check():
#     invalid_array = [[1, 0, 3, 4], [2, 0, 5, 3]]
#     valid_array = np.array(invalid_array)

#     with pytest.raises(TypeError) as exc_info:
#         sanity_check(
#             "my_test_function",
#             ("params", invalid_array, funcs.check_parameter_array, (1,)),
#         )
#     print(str(exc_info.value))
#     sanity_check(
#         "my_test_function",
#         ("params", valid_array, funcs.check_parameter_array, (1,)),
#         ("expinfo", ExpInfo(sw=5000, offset=2000, dim=2), funcs.check_expinfo, ())
#     )

def test():
    sanity_check(
        (
            "hello", [3.0, 5.0], funcs.check_float_list, (),
            {"length": 2, "must_be_positive": True},
        ),
    )
