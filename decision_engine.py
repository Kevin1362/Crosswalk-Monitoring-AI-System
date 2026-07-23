def evaluate_risk(vehicles, pedestrians):

    if pedestrians > 0 and vehicles > 0:
        return "WARNING"

    elif vehicles > 2:
        return "VIOLATION"

    else:
        return "SAFE"