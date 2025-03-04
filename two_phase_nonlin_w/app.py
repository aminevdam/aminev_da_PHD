import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from two_phase import TwoPhase, k_oil, k_water, find_peaks_in_ts
import math

with open("two_phase_nonlin_w/exp_res/solver1.pkl", "rb") as f:
    solver1 = pickle.load(f)
with open("two_phase_nonlin_w/exp_res/solver2.pkl", "rb") as f:
    solver2 = pickle.load(f)
with open("two_phase_nonlin_w/exp_res/solver3.pkl", "rb") as f:
    solver3 = pickle.load(f)
with open("two_phase_nonlin_w/exp_res/solver4.pkl", "rb") as f:
    solver4 = pickle.load(f)
with open("two_phase_nonlin_w/exp_res/solver5.pkl", "rb") as f:
    solver5 = pickle.load(f)
with open("two_phase_nonlin_w/exp_res/solver6.pkl", "rb") as f:
    solver6 = pickle.load(f)

solvers = [solver1, solver2, solver3, solver4, solver5, solver6]

def expit(x):
    try:
        return 1 / (1 + np.exp(-x))
    except:
        return 0.

def calculate_grid(n):
    """
    Вычисляет оптимальное количество строк и столбцов для размещения n графиков.
    """
    cols = math.ceil(math.sqrt(n))  # Количество столбцов — ближайшее целое вверх от корня из n
    rows = math.ceil(n / cols)      # Количество строк — минимальное число строк для размещения всех графиков
    return rows, cols

def generate_data(grad, solver):
    mu_oil = (solver.mu_h-solver.mu_o) * expit (solver1.glad * (-grad + solver.G)) + solver.mu_o
    mu_water_f = (solver.mu_h_w-solver.mu_o_w) * expit (solver.glad_w * (-grad + solver.G_w)) + solver.mu_o_w
    mu_water_b = (solver.mu_h_w_b-solver.mu_o_w) * expit (solver.glad_w_b * (-grad + solver.G_w_b)) + solver.mu_o_w
    return mu_oil, mu_water_f, mu_water_b

def generate_flow_data(sx, solver):
    sx = 20*sx
    sx = np.abs(solver.x*solver.L - sx).argmin()
    dt = solver.dt*solver.T
    grad_p = solver.grad_p*solver.p_0/solver.L
    w_water = -solver.k*k_water(solver.s)/solver.mu_water_arr*grad_p
    w_oil = -solver.k*k_oil(solver.s)/solver.mu_oil_arr*grad_p

    Q_in = w_water[:, 0]*solver.S
    Qw_out = w_water[:, sx]*solver.S
    Qo_out = w_oil[:, sx]*solver.S

    Q_in_sum = [0]
    Qw_out_sum = [0]
    Qo_out_sum = [0]
    for q_in_w, q_o, q_w in zip(Q_in, Qo_out, Qw_out):
        Q_in_sum.append(Q_in_sum[-1]+q_in_w*dt)
        Qw_out_sum.append(Qw_out_sum[-1]+q_w*dt)
        Qo_out_sum.append(Qo_out_sum[-1]+q_o*dt)
    V_in_sum = [0]
    for q_in_w in Q_in:
        V_in_sum.append(V_in_sum[-1]+q_in_w*dt)
    V_in_sum = [i/(solver.S*solver.L*solver.m0*20/solver.L) for i in V_in_sum]
    # V_in_sum = [q*solver.dt/(solver.S*solver.L*solver.m0*20/solver.L) for q in Q_in_sum]
    grad_p = grad_p[:, sx]
    time = solver.time
    mu_oil_arr = solver.mu_oil_arr[:, sx]
    mu_water_arr = solver.mu_water_arr[:, sx]

    return V_in_sum, Qw_out, Qo_out, Qw_out_sum, Qo_out_sum, mu_oil_arr, mu_water_arr, time, grad_p, sx


# Настройки Streamlit
st.info("Сравнение результатов решения задачи двухфазной фильтрации воды и раствора 'полимера' при различных зависимостях вязкости раствора от градиента давления")

st.info("Ниже привдены графики, демонстрирующие зависимость вязкости фаз от градиента давления. 'полимер' обозначается как water с двумя приставками forward - падения вязкости, backward - увеличение вязкости.")

# Флаги для выбора графиков
options = {
    "Закон 1": st.checkbox("Закон 1", False),
    "Закон 2": st.checkbox("Закон 2", False),
    "Закон 3": st.checkbox("Закон 3", False),
    "Закон 4": st.checkbox("Закон 4", False),
    "Закон 5": st.checkbox("Закон 5", False),
    "Закон 6": st.checkbox("Закон 6", False),
}

st.info('Для рсавнения кривых вытеснения при разных зависимостях вязкости от градиента давления необходимо сделать активными соответствующие флаги.')

mapping = {
    "Закон 1": solver1,
    "Закон 2": solver2,
    "Закон 3": solver3,
    "Закон 4": solver4,
    "Закон 5": solver5,
    "Закон 6": solver6,
    }

# Построение графиков 2x3
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
graph_titles = list(options.keys())

grad = np.linspace(0.2e7, 0.7e7, 200)

for i, ax in enumerate(axes.flat):
    mu_oil, mu_water_f, mu_water_b = generate_data(grad, solvers[i])
    ax.plot(grad, mu_oil, label="oil")
    ax.plot(grad, mu_water_f, label="water forward")
    ax.plot(grad, mu_water_b, label="water backward")
    ax.set_xlabel("Градиент давления (Па/м)")
    ax.set_ylabel("Вязкость (Па·с)")
    ax.set_title(graph_titles[i])
    ax.legend()

st.pyplot(fig)

# Отображение выбранных графиков на одном рисунке
selected_graphs = [key for key, value in options.items() if value]
if selected_graphs:
    fig_combined, ax_combined = plt.subplots(figsize=(10, 6))
    for i, title in enumerate(selected_graphs):
        solver = mapping[title]
        mu_oil, mu_water_f, mu_water_b = generate_data(grad, solver)
        # ax_combined.plot(grad, mu_oil, label=f"{title} - нефть")
        ax_combined.plot(grad, mu_water_f, label=f"{title} - вода вперед")
        ax_combined.plot(grad, mu_water_b, label=f"{title} - вода назад")
    ax_combined.set_title("Выбранные графики")
    ax_combined.legend()
    st.pyplot(fig_combined)

# Ползунок для управления масштабом
factor = st.slider("Выберите долю от длины образца, в которой вывести графики", 0.0, 1.0, 0.6, step=0.2)

if selected_graphs:
    fig_dynamic, axes_dynamic = plt.subplots(2, 2, figsize=(30, 22))
    for i, title in enumerate(selected_graphs):
        solver = mapping[title]
        idx = title.split()[1]
        V_in_sum, Qw_out, Qo_out, Qw_out_sum, Qo_out_sum, mu_oil_arr, mu_water_arr, time, grad_p, sx = generate_flow_data(factor, solver)
        # График накопленного объема добычи
        axes_dynamic[0, 0].plot(V_in_sum, Qw_out_sum, '--', label=f'Вода {idx}')
        axes_dynamic[0, 0].plot(V_in_sum, Qo_out_sum, '--', label=f'Нефть {idx}')
        axes_dynamic[0, 0].set_title("Накопленный объем добычи")
        axes_dynamic[0, 0].set_xlabel("Прокаченный поровый объем")
        axes_dynamic[0, 0].set_ylabel("Накопленный объем (м3)")
        axes_dynamic[0, 0].legend()

        # График динамики добычи воды
        axes_dynamic[0, 1].plot(time, Qw_out, '--', label=f'Вода {idx}')
        axes_dynamic[0, 1].plot(time, Qo_out, '--', label=f'Нефть {idx}')
        axes_dynamic[0, 1].set_title("Динамика добычи")
        axes_dynamic[0, 1].set_xlabel("Время безразмерное")
        axes_dynamic[0, 1].set_ylabel("Расход через сечение образца (м^3/c)")
        axes_dynamic[0, 1].legend()

        # График изменения градиента
        axes_dynamic[1, 0].plot(time, grad_p, '--', label=f'Градиент {idx}')
        axes_dynamic[1, 0].set_title("Изменение градиента")
        axes_dynamic[1, 0].set_xlabel("Время безразмерное")
        axes_dynamic[1, 0].set_ylabel("Градиент давления (Па/м)")
        axes_dynamic[1, 0].legend()

        # График изменения вязкости
        axes_dynamic[1, 1].plot(time[1:], mu_oil_arr[1:], '--', label=f'Нефть {idx}')
        axes_dynamic[1, 1].plot(time[1:], mu_water_arr[1:], '--', label=f'Вода {idx}')
        axes_dynamic[1, 1].set_title("Изменение вязкости фаз")
        axes_dynamic[1, 1].set_xlabel("Время безразмерное")
        axes_dynamic[1, 1].set_ylabel("Вязкость (Па·с)")
        axes_dynamic[1, 1].legend()

    st.pyplot(fig_dynamic)

factor1 = st.slider("Выберите интервал времени, где наблюдаются колебания", 0.0, 1.0, value=(0.5, 1.), step=0.05)

if selected_graphs:
    fig_dynamic, axes_dynamic = plt.subplots(len(selected_graphs), 1, figsize=(30, 22))
    for i, title in enumerate(selected_graphs):
        if isinstance(axes_dynamic, np.ndarray):
            ax1 = axes_dynamic[i]
        else:
            ax1 = axes_dynamic
        solver = mapping[title]
        idx = title.split()[1]
        V_in_sum, Qw_out, Qo_out, Qw_out_sum, Qo_out_sum, mu_oil_arr, mu_water_arr, time, grad_p, sx = generate_flow_data(factor, solver)
        
        t0, t1 = factor1
        time_part = (solver.time >= t0) & (solver.time <= t1)
        peaks, values = find_peaks_in_ts(solver.time[time_part], Qw_out[time_part])
        
        # Первая ось (слева)
        color = 'tab:blue'
        ax1.set_xlabel('Время (с)')
        ax1.set_ylabel('Объем добычи м3', color=color)
        ax1.plot(solver.time[time_part], Qw_out[time_part], color=color, label=f'Вода {idx}')
        ax1.plot(solver.time[time_part], Qo_out[time_part], color='tab:orange', label=f'Нефть {idx}')
        for peak in peaks:
            ax1.axvline(solver.time[time_part][peak], color='gray', linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_title(f"Колебания для закона {idx}")

        # Вторая ось (слева, с небольшим смещением)
        ax2 = ax1.twinx()  # Дублируем ось Y
        color2 = 'tab:green'
        ax2.spines['left'].set_position(('axes', -0.1))  # Смещение оси внутрь графика
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()
        ax2.set_ylabel('Вязкость воды Па*с', color=color2)
        ax2.plot(solver.time[time_part], solver.mu_water_arr[time_part, sx], color=color2, linestyle='--', label='Вязкость')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Вторая ось (справа)
        ax3 = ax1.twinx()  # Создаём вторую ось, разделяющую ту же ось x
        color = 'tab:red'
        ax3.set_ylabel('Градиент Па/м', color=color)
        ax3.plot(solver.time[time_part], abs(grad_p[time_part]), color=color, linestyle='-.', label='Градиент')
        ax3.tick_params(axis='y', labelcolor=color)

        # Добавление заголовка и легенды
        fig_dynamic.suptitle('График с двумя вертикальными осями')
        fig_dynamic.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

        # Отображение сетки и графика
        plt.grid(True)
    st.pyplot(fig_dynamic)

st.info(
    """
    ### Анализ влияния параметров системы на характеристики колебаний  
    **w** — вода, **o** — нефть  

    #### 📌 Параметры системы:
    1️⃣ **mu_otn** — отношение mu_h_w (верхняя полка при прямом проходе) к mu_h_w_b (верхняя полка при обратном проходе)  
    2️⃣ **mu_o_w** — нижняя полка в законе для вязкости воды  
    3️⃣ **glad_w** — скорость изменения вязкости в окрестности критического значения градиента при прямом проходе  
    4️⃣ **glad_w_b** — скорость изменения вязкости в окрестности критического значения градиента при обратном проходе  
    5️⃣ **Q** — расход на левой границе керна  

    #### 🎯 Характеристики колебаний:
    1️⃣ **A_(w/o)** — начальная амплитуда колебаний для расхода через сечение со временем  
    2️⃣ **alfa_(w/o)** — коэффициент затухания колебаний:  
       alfa = - (1 / delta_t) * np.log(A_end / A_start)  
    3️⃣ **freq_(w/o)** — доминирующая частота в преобразовании Фурье  
    """
)

st.info("1) График отображает коэффициент корреляции Пирсона, характеризующий линейнуй зависимость между двумя величинами")
st.latex(r"""
r = \frac{\sum (X_i - \bar{X}) (Y_i - \bar{Y})}
{\sqrt{\sum (X_i - \bar{X})^2} \cdot \sqrt{\sum (Y_i - \bar{Y})^2}}
""")
st.image("two_phase_nonlin_w/exp_res/corr.png")

st.info("2) Представление парных диаграмм рассеяния")
st.image("two_phase_nonlin_w/exp_res/plots.png")

st.info("3) Оценка важности параметров в задаче поиска отображения из пространства параметров в пространство характеристик колебаний. Оценка показывает, насколько каждый параметр влияет на точность отображения.")
st.image("two_phase_nonlin_w/exp_res/model.png")
