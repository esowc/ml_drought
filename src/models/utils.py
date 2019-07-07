import torch
import numpy as np
from random import shuffle as shuffle_list
from typing import cast, Iterable, Union, Tuple


def chunk_array(x: Union[Tuple[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
                y: Union[torch.Tensor, np.ndarray],
                batch_size: int,
                shuffle: bool = False) -> Iterable[Tuple[Tuple[Union[torch.Tensor, np.ndarray]],
                                                         Union[torch.Tensor, np.ndarray]]]:
    """
    Chunk an array into batches of batch size `batch_size`

    Arguments
    ----------
    x: ({torch.Tensor, np.ndarray})
        The x tensors to chunk
    y: {torch.Tensor, np.ndarray}
        The y tensor to chunk. Must be the same type as x
    batch_size: int
        The size of the batches to return
    shuffle: bool = False
        Whether to shuffle the returned tensors

    Returns
    ----------
    An iterator returning tuples of batches (x, y)
    """
    if type(x) is not tuple:
        x = (x, )
    x = cast(Tuple[Union[torch.Tensor, np.ndarray]], x)

    num_sections = max(1, x[0].shape[0] // batch_size)
    if type(x[0]) == np.ndarray:
        return _chunk_ndarray(x, y, num_sections, shuffle)
    else:
        return _chunk_tensor(x, y, num_sections, shuffle)


def _chunk_ndarray(x: Tuple[np.ndarray], y: np.ndarray,
                   num_sections: int,
                   shuffle: bool) -> Iterable[Tuple[Tuple[np.ndarray], np.ndarray]]:

    split_x = [np.array_split(x_section, num_sections) for x_section in x]
    split_y = np.array_split(y, num_sections)
    return_arrays = list(zip(*split_x, split_y))

    if shuffle:
        shuffle_list(return_arrays)
    return [(chunk[:-1], chunk[-1]) for chunk in return_arrays]  # type: ignore


def _chunk_tensor(x: Tuple[torch.Tensor], y: torch.Tensor,
                  num_sections: int,
                  shuffle: bool) -> Iterable[Tuple[Tuple[torch.Tensor], torch.Tensor]]:
    split_x = [torch.chunk(x_section, num_sections) for x_section in x]
    split_y = torch.chunk(y, num_sections)
    return_arrays = list(zip(*split_x, split_y))

    if shuffle:
        shuffle_list(return_arrays)
    return [(chunk[:-1], chunk[-1]) for chunk in return_arrays]  # type: ignore
