def inner_fn():
    a = [i * i for i in range(1000)]
    return a


# ひどいコード
def fn(length: int = 1000) -> list:
    """sample

    Args:
        length: array length. Defaults to 1000.

    Returns:
        list: array
    """
    a = []
    for i in range(length):
        a.append(i * i)
    return inner_fn()
