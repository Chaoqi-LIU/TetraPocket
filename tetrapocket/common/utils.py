import signal
import sys
from typing import Optional, Callable



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


def wait_user_input(
    valid_input: Callable[[str], bool],
    prompt: str = '',
    default: str = '',
    continue_after: Optional[int] = None,
) -> str:
    # valid_input: check if user input is valid
    # default: default value if user input is empty
    # continue_after: if not None, pause for a while before return the default value
    # return: the user's input

    class TimeoutExpired(Exception):
        pass

    def timeout_handler(*args, **kwargs):
        raise TimeoutExpired
    
    if continue_after is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(continue_after))

    try:
        while not valid_input(user_input := input(prompt)):
            print('Invalid input') 
    except TimeoutExpired:
        user_input = ''

    if user_input == '':
        user_input = default
    return user_input
            
