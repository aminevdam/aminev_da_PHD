import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pickle
from two_phase import TwoPhase, k_oil, k_water

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
    time = solver.time[1:]
    mu_oil_arr = solver.mu_oil_arr[1:, sx]
    mu_water_arr = solver.mu_water_arr[1:, sx]

    return V_in_sum, Qw_out[1:], Qo_out[1:], Qw_out_sum, Qo_out_sum, mu_oil_arr, mu_water_arr, time, grad_p[1:], sx


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
factor = st.slider("Выберите долю от длины образца, в которой вывести графики", 0.0, 1.0, 0.1, step=0.2)

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
        axes_dynamic[1, 1].plot(time, mu_oil_arr, '--', label=f'Нефть {idx}')
        axes_dynamic[1, 1].plot(time, mu_water_arr, '--', label=f'Вода {idx}')
        axes_dynamic[1, 1].set_title("Изменение вязкости фаз")
        axes_dynamic[1, 1].set_xlabel("Время безразмерное")
        axes_dynamic[1, 1].set_ylabel("Вязкость (Па·с)")
        axes_dynamic[1, 1].legend()

    st.pyplot(fig_dynamic)
