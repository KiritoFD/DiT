"""Cross-platform stand-in for POSIX fcntl: no-op locking on Windows."""
try:
    import fcntl as _posix

    LOCK_EX = _posix.LOCK_EX
    LOCK_NB = _posix.LOCK_NB
    LOCK_SH = _posix.LOCK_SH

    def flock(fd, op):
        return _posix.flock(fd, op)

except ImportError:  # Windows
    LOCK_EX = LOCK_SH = LOCK_NB = 0

    def flock(fd, op):
        """No advisory locking on Windows; callers already do serial execution."""
        return None
