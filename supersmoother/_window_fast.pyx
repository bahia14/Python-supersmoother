import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int64
ctypedef np.int64_t ITYPE_t

cdef inline ITYPE_t imax(ITYPE_t a, ITYPE_t b): return a if a >= b else b
cdef inline ITYPE_t imin(ITYPE_t a, ITYPE_t b): return a if a <= b else b


@cython.cdivision(True)
cdef inline ITYPE_t floordiv(ITYPE_t a, ITYPE_t b):
    return a // b if a >= 0 else (a + 1 - b) // b


@cython.cdivision(True)
cdef inline ITYPE_t imod(ITYPE_t a, ITYPE_t b):
    cdef ITYPE_t res = a % b
    return res if res >= 0 else res + b


def windowed_sum(arrays, span, t=None, indices=None, tpowers=0,
                 period=None, subtract_mid=False):
    """Compute the windowed sum of the given arrays.

    This is a fast version based on cython code.

    Parameters
    ----------
    arrays : tuple of arrays
        arrays to window
    span : int or array of int
        The span to use for the sum at each point. If array is provided,
        it must be broadcastable with ``indices``
    indices : array
        the indices of the center of the desired windows. If ``None``,
        the indices are assumed to be ``range(len(arrays[0]))`` though
        these are not actually instantiated.
    t : array (optional)
        Times associated with the arrays
    tpowers : list (optional)
        Powers of t for each array sum
    period : float (optional)
        Period to use, if times are periodic. If supplied, input times
        must be arranged such that (t % period) is sorted!
    subtract_mid : boolean
        If true, then subtract the middle value from each sum

    Returns
    -------
    arrays : tuple of ndarrays
        arrays containing the windowed sum of each input array
    """
    from ._window_fast import (_window_fixed, _window_fixed_periodic,
                               _window_variable, _window_variable_periodic,
                               DTYPE, ITYPE)
    span = np.asarray(span, dtype=ITYPE)
    if not np.all(span > 0):
        raise ValueError("span values must be positive")
    
    arrays = tuple(np.asarray(a, dtype=DTYPE, order='C') for a in arrays)
    N = arrays[0].size
    if not all(a.shape == (N,) for a in arrays):
        raise ValueError("sizes of provided arrays must match")
    
    t_input = t
    if t is not None:
        t = np.asarray(t, dtype=DTYPE, order='C')
        if t.shape != (N,):
            raise ValueError("shape of t must match shape of arrays")
    else:
        t = np.ones(N, dtype=DTYPE)
    
    tpowers = tpowers + np.zeros(len(arrays), dtype=int)
    if len(tpowers) != len(arrays):
        raise ValueError("tpowers must be broadcastable with number of arrays")

    if period:
        if t_input is None:
            raise ValueError("periodic requires t to be provided")
        t = t % period
    
    if indices is not None:
        span, indices = np.broadcast_arrays(span, indices)
    
    results = []
    for tpower, array in zip(tpowers, arrays):
        if span.ndim == 0:
            # fixed-span case.
            assert indices is None
            if period:
                result = _window_fixed_periodic(array, t, tpower,
                                                span, subtract_mid, period)
            else:
                array = np.asarray(array * t ** tpower,
                                   dtype=DTYPE, order='C')
                result = _window_fixed(array, span, subtract_mid)
        else:
            # variable span case.
            if indices is None:
                indices = np.arange(len(span), dtype=ITYPE)

            span = np.asarray(span, dtype=ITYPE, order='C')
            indices = np.asarray(indices, dtype=ITYPE, order='C')
            assert span.shape == indices.shape

            if period:
                result = _window_variable_periodic(array, t, tpower, indices,
                                                   span, subtract_mid, period)
            else:
                array = np.asarray(array * t ** tpower,
                                   dtype=DTYPE, order='C')
                result = _window_variable(array, indices, span, subtract_mid)
        results.append(np.asarray(result))
    return tuple(results)


#@cython.boundscheck(False)
@cython.cdivision(True)
def _window_fixed(DTYPE_t[::1] a, ITYPE_t span, bint subtract_mid):
    cdef ITYPE_t i, j, start, N
    N = a.shape[0]
    cdef DTYPE_t[::1] result = np.zeros(N, dtype=DTYPE, order='C')
    for i in range(N):
        start = i - span // 2
        for j in range(imax(0, start), imin(N, start + span)):
            if subtract_mid and j == i:
                continue
            result[i] += a[j]
    return result


#@cython.boundscheck(False)
@cython.cdivision(True)
def _window_fixed_periodic(DTYPE_t[::1] a, DTYPE_t[::1] t, int tpower,
                           ITYPE_t span, bint subtract_mid, DTYPE_t period):
    cdef ITYPE_t i, j, start, N
    N = a.shape[0]
    cdef DTYPE_t[::1] result = np.zeros(N, dtype=DTYPE, order='C')
    for i in range(N):
        start = i - span // 2
        for j in range(start, start + span):
            if subtract_mid and j == i:
                continue
            result[i] += (a[imod(j, N)] *
                          (t[imod(j, N)] + floordiv(j, N) * period) ** tpower)
    return result


#@cython.boundscheck(False)
@cython.cdivision(True)
def _window_variable(DTYPE_t[::1] a, ITYPE_t[::1] ind,
                     ITYPE_t[::1] span, bint subtract_mid):
    cdef ITYPE_t i, j, start, N, M
    N = a.shape[0]
    M = span.shape[0]
    cdef DTYPE_t[::1] result = np.zeros(M, dtype=DTYPE, order='C')
    for i in range(M):
        start = ind[i] - span[i] // 2
        for j in range(imax(0, start), imin(N, start + span[i])):
            if subtract_mid and j == ind[i]:
                continue
            result[i] += a[j]
    return result


#@cython.boundscheck(False)
@cython.cdivision(True)
def _window_variable_periodic(DTYPE_t[::1] a, DTYPE_t[::1] t, int tpower,
                              ITYPE_t[::1] ind, ITYPE_t[::1] span,
                              bint subtract_mid, DTYPE_t period):
    cdef ITYPE_t i, j, start, N, M
    N = a.shape[0]
    M = span.shape[0]
    cdef DTYPE_t[::1] result = np.zeros(M, dtype=DTYPE, order='C')
    for i in range(M):
        start = ind[i] - span[i] // 2
        for j in range(start, start + span[i]):
            if subtract_mid and j == ind[i]:
                continue
            result[i] += (a[imod(j, N)] *
                          (t[imod(j, N)] + floordiv(j, N) * period) ** tpower)
    return result


@cython.cdivision(True)
def test_div(int a, int b):
    return (floordiv(a, b), imod(a, b))
