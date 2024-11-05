import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import math

def solution_plot(data, titles, idx, nt, height, width):
    # data: List[List], titles: List[str]

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