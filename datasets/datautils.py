"""
	Utils functions to deal with spherical coordinates in Pytorch.
"""

from math import pi
import torch
import os
import numpy as np
from numpy.linalg import norm


def circular_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])
    v1 = normalize(v1)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    pos_rcv *= radius
    return pos_rcv


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


def audiowu_high_array_geometry() -> np.array:
    # the high-resolution mic array of the audio lab of westlake university
    R = 0.03
    pos_rcv = np.zeros((32, 3))
    pos_rcv[1:9, :] = circular_array_geometry(radius=R, mic_num=8)
    pos_rcv[9:17, :] = circular_array_geometry(radius=R * 2, mic_num=8)
    pos_rcv[17:25, :] = circular_array_geometry(radius=R * 3, mic_num=8)
    pos_rcv[25, :] = np.array([-R * 4, 0, 0])
    pos_rcv[26, :] = np.array([R * 4, 0, 0])
    pos_rcv[27, :] = np.array([R * 5, 0, 0])

    L = 0.045
    pos_rcv[28, :] = np.array([0, 0, L * 2])
    pos_rcv[29, :] = np.array([0, 0, L])
    pos_rcv[30, :] = np.array([0, 0, -L])
    pos_rcv[31, :] = np.array([0, 0, -L * 2])
    return pos_rcv


def search_files(dir_path, flag):
    result = []
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name)
        if os.path.isdir(complete_file_name):
            result.extend(search_files(complete_file_name, flag))
        if os.path.isfile(complete_file_name):
            if complete_file_name.endswith(flag):
                result.append(complete_file_name)
    return result


def angular_error(the_pred, phi_pred, the_true, phi_true):
    """ Angular distance between spherical coordinates.
    """
    aux = torch.cos(the_true) * torch.cos(the_pred) + \
          torch.sin(the_true) * torch.sin(the_pred) * torch.cos(phi_true - phi_pred)

    return torch.acos(torch.clamp(aux, -0.99999, 0.99999))
