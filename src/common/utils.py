def sequential_transforms(*transforms):
    """

    Parameters
    ----------
    transforms

    Returns
    -------

    """
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func