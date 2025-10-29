def enable_autoreload():
    """Enable autoreload if running in IPython/Jupyter."""
    try:
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is not None:
            ipy.run_line_magic("load_ext", "autoreload")
            ipy.run_line_magic("autoreload", "2")
            print("Edit mode: autoreload enabled")
    except Exception as e:
        print(f"Failed to enable autoreload: {e}")