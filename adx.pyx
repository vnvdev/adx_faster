import numpy as np
cimport numpy as np
cimport cython

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ExponentialMA(int i, int period, double prev_value, np.ndarray[np.double_t, ndim=1] values):
    cdef double ema
    if i == 0:
        return prev_value
    else:
        ema = (values[i] - prev_value) * 2 / (period + 1) + prev_value
        return ema

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef ADX(np.ndarray[np.double_t, ndim=1] high, np.ndarray[np.double_t, ndim=1] low, np.ndarray[np.double_t, ndim=1] close, int adx_period):
    cdef int i
    cdef double high_price, prev_high, low_price, prev_low, prev_close, tmp_pos, tmp_neg, tr, tmp
    cdef np.ndarray[np.double_t, ndim=1] pdi = np.zeros(high.shape[0])
    cdef np.ndarray[np.double_t, ndim=1] ndi = np.zeros(high.shape[0])
    cdef np.ndarray[np.double_t, ndim=1] adx = np.zeros(high.shape[0])
    cdef np.ndarray[np.double_t, ndim=1] tmp_buffer = np.zeros(high.shape[0])

    for i in range(1, len(high)):
        high_price = high[i]
        prev_high = high[i-1]
        low_price = low[i]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        tmp_pos = high_price - prev_high
        tmp_neg = prev_low - low_price
        if tmp_pos < 0:
            tmp_pos = 0
        if tmp_neg < 0:
            tmp_neg = 0
        if tmp_pos > tmp_neg:
            tmp_neg = 0
        else:
            if tmp_pos < tmp_neg:
                tmp_pos = 0
            else:
                tmp_pos = 0
                tmp_neg = 0
        
        tr = max(high_price - low_price, abs(high_price - prev_close), abs(low_price - prev_close))
        
        pdi[i] = 100 * tmp_pos / tr if tr != 0 else 0
        ndi[i] = 100 * tmp_neg / tr if tr != 0 else 0
        
        pdi[i] = ExponentialMA(i, adx_period, pdi[i-1], pdi)
        ndi[i] = ExponentialMA(i, adx_period, ndi[i-1], ndi)
        tmp = pdi[i] + ndi[i]
        tmp_buffer[i] = 100 * abs((pdi[i] - ndi[i]) / tmp) if tmp != 0 else 0
    
    for i in range(1, len(high)):
        adx[i] = ExponentialMA(i, adx_period, adx[i-1], tmp_buffer)
    
    return adx

        
