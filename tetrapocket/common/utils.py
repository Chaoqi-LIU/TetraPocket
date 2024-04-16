import torch
import sys
from typing import (
    Optional,
)



def sec2hms(sec: float) -> str:
    """
    Convert seconds to hours, minutes and seconds.
    :param sec: seconds
    :return: str, f'{hours}:{minutes}:{seconds}'
    """
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f'{int(h)}:{int(m)}:{s:.2f}'


def print_progress_bar(
    iteration: float, 
    total: float, 
    prefix: Optional[str] = '', 
    suffix: Optional[str] = '', 
    length: Optional[int] = 50, 
    fill: Optional[str] = '█'
) -> None:
    """
    Print progress bar.
    :param iteration: current iteration
    :param total: total iterations
    :param prefix: prefix
    :param suffix: suffix
    :param length: length of progress bar
    :param fill: fill character
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * ((length - filled_length)-1)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()


