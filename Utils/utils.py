import psutil
class VerilogExecutionError(Exception):
    def __init__(self, message,error_type):
        super().__init__(error_type)
        self.type=error_type
        self.error_message=message

def kill_processes(active_processes):
    for process in active_processes:
        if process.poll() is None:  
            kill_process_tree(process.pid)
    active_processes.clear()  

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):  
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass  
    except:
        pass