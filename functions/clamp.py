def clamp(minimum: int, x: int, maximum: int):
    """Clamps an integer between a min/max"""
    return max(minimum, min(x, maximum))
