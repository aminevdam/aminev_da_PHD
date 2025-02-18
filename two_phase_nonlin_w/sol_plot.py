import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from math import ceil

def solution_plot(data, titles, idx, nt, height, width, fraction=1.0):
    """
    Строит график с ползунком для нескольких наборов данных.
    
    Args:
        data (list): Список списков, где каждый элемент — это [x, [y_1, y_2, ..., y_nt]].
        titles (list): Заголовки для каждого графика.
        idx (list): Индексы положения графиков в сетке subplot.
        nt (int): Количество временных шагов.
        height (int): Высота графика.
        width (int): Ширина графика.
        fraction (float): Доля данных для отображения (0 < fraction <= 1.0).
    """

    # Учитываем долю данных
    if fraction <= 0 or fraction > 1.0:
        raise ValueError("fraction должен быть в диапазоне (0, 1].")
    
    # Пропускаем данные с равным шагом
    def sample_data(array, fraction):
        n_samples = ceil(len(array) * fraction)
        indices = np.linspace(0, len(array)-1, n_samples, dtype=int)
        res = np.array([[array[i]] for i in indices]).reshape(-1, array.shape[-1])
        return res

    data = [
        [dp[0], sample_data(dp[1], fraction)] for dp in data]

    nt = data[0][1].shape[0]

    # Определение количества строк и столбцов для subplot
    rows = max([id[0] for id in idx])
    cols = max([id[1] for id in idx])

    # Создаем subplot для размещения двух графиков
    fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True,
                    subplot_titles=titles)
    
    # Создаем пустые графики для p и s
    plots_0 = [go.Scatter(x=dp[0], y=dp[1][0], mode='lines') for dp in data]

    for i, plot in enumerate(plots_0):
        fig.add_trace(plot, row=idx[i][0], col=idx[i][1])

    # Определяем шаги ползунка, где каждый шаг обновляет данные для каждого момента времени
    steps = []
    for i in range(nt):
        y = [dp[1][i] for dp in data]
        step = dict(
            method="update",
            args=[{"y": y},
                {"title": f"Time step: {i}"}],  # обновление заголовка
            label=f"{i}"
        )
        steps.append(step)

    # Добавляем ползунок (slider)
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time step: "},
        pad={"t": 50},
        steps=steps
    )]

    # Настраиваем внешний вид графиков
    fig.update_layout(
        sliders=sliders,
        height=height,
        width=width,
        showlegend=False,
    )

    # Отображаем график
    fig.show()
