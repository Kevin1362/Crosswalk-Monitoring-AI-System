def evaluate_state(vehicles, pedestrians):
    """
    UC-1 decision logic
    """

    if vehicles > 0 and pedestrians > 0:
        return "VIOLATION"

    elif vehicles > 0 and pedestrians == 0:
        return "WARNING"

    else:
        return "SAFE"