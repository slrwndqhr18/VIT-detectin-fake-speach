from multiprocessing import Process, Manager
from functools import wraps
import os

def Multi_PS_wrapper(_func):
    #def Wrapper(_pid, _params, _retManager):
    @wraps(_func)
    def Wrapper(_pid, _params, _retManager):
        try:
            res = _func(_params)
            _retManager[_pid] = {"STAT": True, "RES": res }
        except Exception as e:
            print(e)
            print("\t\tError at PID",_pid)
            _retManager[_pid] = {"STAT": False}
        return
    return Wrapper

def Exec_in_parallel(_func, _psArgs, _psCnt=None):
    if _psCnt == None:
        _psCnt = os.cpu_count()
    manager = Manager()
    psResult = manager.dict()

    PS = []
    for pid in range(len(_psArgs)):
        p = Process(target=_func, args=(pid, _psArgs[pid], psResult))
        PS.append(p)
        p.start()
    for p in PS:
        p.join()
    return psResult.values()