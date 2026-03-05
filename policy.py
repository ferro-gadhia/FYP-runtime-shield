##checks if the system policy is safe or unsafe

def can_read_file(path: str) -> bool:
    if path.startswith("/sandbox/"):
        return True
    #secret file are forbidden
    if path.startswith("/secret/"):
        return False
    return False

def can_write_file(path: str) -> bool:
    #returns true if writing to path
    if path.startswith("/secret/"):
        return False

    if path.startswith("/sandbox/") or path.startswith("/exfil/"):
        return True
    
    return False

def network_range_check(network_new: str, range: str) -> bool:
    return range == network_new
    