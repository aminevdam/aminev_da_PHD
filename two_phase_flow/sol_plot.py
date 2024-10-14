import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def solution_plot(data, titles):
    # data: List[List], titles: List[str]

    rows = len(data)

    # Создаем subplot для размещения двух графиков
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                    subplot_titles=titles)
    

# Создаем пустые графики для p и s
trace_p = go.Scatter(x=solver.x, y=p[0], mode='lines', name='Pressure (p)')
trace_s = go.Scatter(x=solver.x, y=s[0], mode='lines', name='Saturation (s)')
trace_s_b = go.Scatter(x=solver.x, y=s_b[0], mode='lines', name='Saturation buckley (s)')
trace_mu_oil = go.Scatter(x=solver.x, y=solver.mu_oil_arr[0], mode='lines', name='mu_oil (s)')
trace_grad_p = go.Scatter(x=solver.x, y=solver.grad_p[0], mode='lines', name='grad (p)')
trace_G = go.Scatter(x=solver.x, y=G[0], mode='lines', name='grad (p)')

# Добавляем их на соответствующие позиции
fig.add_trace(trace_p, row=1, col=1)
fig.add_trace(trace_s, row=2, col=1)
fig.add_trace(trace_s_b, row=2, col=1)
fig.add_trace(trace_mu_oil, row=3, col=1)
fig.add_trace(trace_grad_p, row=4, col=1)
fig.add_trace(trace_G, row=4, col=1)
# Определяем шаги ползунка, где каждый шаг обновляет данные для каждого момента времени
steps = []
for i in range(solver.nt):
    step = dict(
        method="update",
        args=[{"y": [p[i], s[i], s_b[i], np.round(solver.mu_oil_arr[i],3), solver.grad_p[i], G[i]]},  # обновление данных для обоих графиков
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
    height=1500,
    width=800,
    showlegend=False,
    title="Interactive Plot for p and s vs x"
)

# Подписи осей
fig.update_xaxes(title_text="x", row=1, col=1)
fig.update_xaxes(title_text="x", row=2, col=1)
fig.update_yaxes(title_text="p", row=1, col=1)
fig.update_yaxes(title_text="s", row=2, col=1)

# Отображаем график
fig.show()
