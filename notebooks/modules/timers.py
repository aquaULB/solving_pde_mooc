from timeit import default_timer
from functools import wraps

def dummy_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t_init = default_timer()
        res = func(*args, **kwargs)
        t_fin = default_timer()
        
        print(f'{func.__name__} executed in {t_fin-t_init} s')
        
        return res
    return wrapper
