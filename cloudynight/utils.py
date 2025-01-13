import time
from functools import wraps

import streamlit as st


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        duration = end - start
        if "timing_logs" not in st.session_state:
            st.session_state.timing_logs = []
        st.session_state.timing_logs.append(f"{func.__name__}: {duration:.2f} seconds")
        return result

    return wrapper
