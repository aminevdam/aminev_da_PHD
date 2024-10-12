def parse_value(value):
    try:
        # Пытаемся вычислить значение с помощью eval (для выражений типа 0.4/86400)
        return float(eval(value))
    except (SyntaxError, NameError):
        # Если это не выражение, возвращаем как float обычное значение
        return float(value)