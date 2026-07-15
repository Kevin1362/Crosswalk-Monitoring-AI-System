"""
UC-2 Safe-State Decision Engine
Responsible for evaluating pedestrian and vehicle risk.
"""


def evaluate_risk(vehicles, pedestrians):
    """
    Determine risk level based on detected objects.

    Parameters:
        vehicles (int): number of vehicles detected
        pedestrians (int): number of pedestrians detected

    Returns:
        str: risk state
    """

    if pedestrians > 0 and vehicles > 0:
        return "HIGH_RISK"

    elif pedestrians > 0:
        return "MEDIUM_RISK"

    else:
        return "SAFE"


def make_decision(risk_level):
    """
    Convert risk level into vehicle action.
    """

    if risk_level == "HIGH_RISK":
        return "HOLD_POSITION"

    elif risk_level == "MEDIUM_RISK":
        return "SLOW_DOWN"

    else:
        return "PROCEED"