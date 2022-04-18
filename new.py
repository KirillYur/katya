import streamlit as st
from iapws import IAPWS97 as WSP
import numpy as np
import pandas as pd
import math as M
import matplotlib.pyplot as plt
from bokeh.plotting import figure
st.write('# Курсовая работа по курсу "Паровые и газовые турбины"')
st.write('# Выполнила: Парнова Екатерина')
Ne = 816e6  # МВт
p0 = 12.5e6  # МПа
t0 = 552  # C
T0 = t0 + 273.15  # K
ppp = 3.74e6  # МПа
tpp = 558  # C
Tpp = tpp + 273.15  # K
pk = list(range(int(2e3), int(10e3), 500))
tpv = 275  # C
Tpv = tpv + 273.15  # K

pk_min = 2e3
pk_max = 10e3

delta_p_0 = 0.05 * p0
delta_p_pp = 0.08 * ppp
delta_p = 0.03 * ppp



def Calculate_G0_Gk(N_e, p_0, T_0, p_pp, T_pp, p_k, T_pv):
    # Потери:
    d_p0 = 0.05
    d_p_pp = 0.1
    d_p = 0.03
    # Параметры свежего пара
    point_0 = IAPWS97(P=p_0 * 10 ** (-6), T=T_0)
    s_0 = point_0.s
    h_0 = point_0.h
    v_0 = point_0.v
    #
    p_0_ = p_0 - 0.05 * p_0
    point_p_0_ = IAPWS97(P=p_0_ * 10 ** (-6), h=h_0)
    t_0_ = point_p_0_.T - 273.15
    s_0_ = point_p_0_.s
    v_0_ = point_p_0_.v
    # Теоретический процесс расширения в ЦВД
    p_1t = p_pp + 0.1 * p_pp
    point_1t = IAPWS97(P=p_1t * 10 ** (-6), s=s_0)
    t_1t = point_1t.T - 273.15
    h_1t = point_1t.h
    v_1t = point_1t.v
    #
    point_pp = IAPWS97(P=p_pp * 10 ** (-6), T=T_pp)
    h_pp = point_pp.h
    s_pp = point_pp.s
    v_pp = point_pp.v
    # Действительный процесс расширения в ЦВД
    H_0 = h_0 - h_1t
    eta_oi = 0.85
    H_i_cvd = H_0 * eta_oi
    h_1 = h_0 - H_i_cvd
    point_1 = IAPWS97(P=p_1t * 10 ** (-6), h=h_1)
    s_1 = point_1.s
    T_1 = point_1.T
    v_1 = point_1.v
    #
    p_pp_ = p_pp - 0.03 * p_pp
    point_pp_ = IAPWS97(P=p_pp_ * 10 ** (-6), h=h_pp)
    s_pp_ = point_pp_.s
    v_pp_ = point_pp_.v
    #
    point_kt = IAPWS97(P=p_k * 10 ** (-6), s=s_pp)
    T_kt = point_kt.T
    h_kt = point_kt.h
    v_kt = point_kt.v
    s_kt = s_pp
    #
    H_0_csdcnd = h_pp - h_kt
    eta_oi = 0.85
    H_i_csdcnd = H_0_csdcnd * eta_oi
    h_k = h_pp - H_i_csdcnd
    point_k = IAPWS97(P=p_k * 10 ** (-6), h=h_k)
    T_k = point_k.T
    s_k = point_k.s
    v_k = point_k.v
    #
    point_k_v = IAPWS97(P=p_k * 10 ** (-6), x=0)
    h_k_v = point_k_v.h
    s_k_v = point_k_v.s
    eta_oiI = (h_1 - h_0) / (h_1t - h_0)
    p_pv = 1.4 * p_0
    point_pv = IAPWS97(P=p_pv * 10 ** (-6), T=T_pv)
    h_pv = point_pv.h
    s_pv = point_pv.s
    #
    ksi_pp_oo = 1 - (1 - (T_k * (s_pp - s_k_v)) / ((h_0 - h_1t) + (h_pp - h_k_v))) / (
                1 - (T_k * (s_pp - s_pv)) / ((h_0 - h_1t) + (h_pp - h_pv)))
    T_0_ = IAPWS97(P=p_pv * 10 ** (-6), x=0).T
    T_ = (point_pv.T - point_k.T) / (T_0_ - point_k.T)
    if T_ <= 0.636364:
        ksi1 = -1.53 * T_ ** 2 + 2.1894 * T_ + 0.0048
    elif 0.636364 < T_ <= 0.736364:
        ksi1 = -1.3855 * T_ ** 2 + 2.00774 * T_ + 0.0321
    elif 0.736364 < T_ <= 0.863636:
        ksi1 = -2.6536 * T_ ** 2 + 4.2556 * T_ - 0.8569

    if T_ <= 0.631818:
        ksi2 = -1.7131 * T_ ** 2 + 2.3617 * T_ - 0.0142
    elif 0.631818 < T_ <= 0.718182:
        ksi2 = -2.5821 * T_ ** 2 + 3.689 * T_ - 0.4825
    elif 0.718182 < T_ <= 0.827273:
        ksi2 = -2.5821 * T_ ** 2 + 3.138 * T_ - 0.3626

    ksi = (ksi1 + ksi2) / 2
    ksi_r_pp = ksi * ksi_pp_oo
    eta_ir = (H_i_cvd + H_i_csdcnd) / (H_i_cvd + (h_pp - h_k_v)) * 1 / (1 - ksi_r_pp)
    H_i = eta_ir * ((h_0 - h_pv) + (h_pp - h_1))
    eta_m = 0.994
    eta_eg = 0.99
    G_0 = N_e / (H_i * eta_m * eta_eg * (10 ** 3))
    G_k = N_e / ((h_k - h_k_v) * eta_m * eta_eg * (10 ** 3)) * (1 / eta_ir - 1)
    return eta_ir


eta = [Calculate_G0_Gk(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk]


def Calculate_G0(N_e, p_0, T_0, p_pp, T_pp, p_k, T_pv):
    # Потери:
    d_p0 = 0.05
    d_p_pp = 0.1
    d_p = 0.03
    # Параметры свежего пара
    point_0 = IAPWS97(P=p_0 * 10 ** (-6), T=T_0)
    s_0 = point_0.s
    h_0 = point_0.h
    v_0 = point_0.v
    #
    p_0_ = p_0 - 0.05 * p_0
    point_p_0_ = IAPWS97(P=p_0_ * 10 ** (-6), h=h_0)
    t_0_ = point_p_0_.T - 273.15
    s_0_ = point_p_0_.s
    v_0_ = point_p_0_.v
    # Теоретический процесс расширения в ЦВД
    p_1t = p_pp + 0.1 * p_pp
    point_1t = IAPWS97(P=p_1t * 10 ** (-6), s=s_0)
    t_1t = point_1t.T - 273.15
    h_1t = point_1t.h
    v_1t = point_1t.v
    #
    point_pp = IAPWS97(P=p_pp * 10 ** (-6), T=T_pp)
    h_pp = point_pp.h
    s_pp = point_pp.s
    v_pp = point_pp.v
    # Действительный процесс расширения в ЦВД
    H_0 = h_0 - h_1t
    eta_oi = 0.85
    H_i_cvd = H_0 * eta_oi
    h_1 = h_0 - H_i_cvd
    point_1 = IAPWS97(P=p_1t * 10 ** (-6), h=h_1)
    s_1 = point_1.s
    T_1 = point_1.T
    v_1 = point_1.v
    #
    p_pp_ = p_pp - 0.03 * p_pp
    point_pp_ = IAPWS97(P=p_pp_ * 10 ** (-6), h=h_pp)
    s_pp_ = point_pp_.s
    v_pp_ = point_pp_.v
    #
    point_kt = IAPWS97(P=p_k * 10 ** (-6), s=s_pp)
    T_kt = point_kt.T
    h_kt = point_kt.h
    v_kt = point_kt.v
    s_kt = s_pp
    #
    H_0_csdcnd = h_pp - h_kt
    eta_oi = 0.85
    H_i_csdcnd = H_0_csdcnd * eta_oi
    h_k = h_pp - H_i_csdcnd
    point_k = IAPWS97(P=p_k * 10 ** (-6), h=h_k)
    T_k = point_k.T
    s_k = point_k.s
    v_k = point_k.v
    #
    point_k_v = IAPWS97(P=p_k * 10 ** (-6), x=0)
    h_k_v = point_k_v.h
    s_k_v = point_k_v.s
    eta_oiI = (h_1 - h_0) / (h_1t - h_0)
    p_pv = 1.4 * p_0
    point_pv = IAPWS97(P=p_pv * 10 ** (-6), T=T_pv)
    h_pv = point_pv.h
    s_pv = point_pv.s
    #
    ksi_pp_oo = 1 - (1 - (T_k * (s_pp - s_k_v)) / ((h_0 - h_1t) + (h_pp - h_k_v))) / (
                1 - (T_k * (s_pp - s_pv)) / ((h_0 - h_1t) + (h_pp - h_pv)))
    T_0_ = IAPWS97(P=p_pv * 10 ** (-6), x=0).T
    T_ = (point_pv.T - point_k.T) / (T_0_ - point_k.T)
    if T_ <= 0.636364:
        ksi1 = -1.53 * T_ ** 2 + 2.1894 * T_ + 0.0048
    elif 0.636364 < T_ <= 0.736364:
        ksi1 = -1.3855 * T_ ** 2 + 2.00774 * T_ + 0.0321
    elif 0.736364 < T_ <= 0.863636:
        ksi1 = -2.6536 * T_ ** 2 + 4.2556 * T_ - 0.8569

    if T_ <= 0.631818:
        ksi2 = -1.7131 * T_ ** 2 + 2.3617 * T_ - 0.0142
    elif 0.631818 < T_ <= 0.718182:
        ksi2 = -2.5821 * T_ ** 2 + 3.689 * T_ - 0.4825
    elif 0.718182 < T_ <= 0.827273:
        ksi2 = -2.5821 * T_ ** 2 + 3.138 * T_ - 0.3626
    ksi = (ksi1 + ksi2) / 2
    ksi_r_pp = ksi * ksi_pp_oo
    eta_ir = (H_i_cvd + H_i_csdcnd) / (H_i_cvd + (h_pp - h_k_v)) * 1 / (1 - ksi_r_pp)
    H_i = eta_ir * ((h_0 - h_pv) + (h_pp - h_1))
    eta_m = 0.994
    eta_eg = 0.99
    G_0 = N_e / (H_i * eta_m * eta_eg * (10 ** 3))
    G_k = N_e / ((h_k - h_k_v) * eta_m * eta_eg * (10 ** 3)) * (1 / eta_ir - 1)
    return G_0


G_0 = [Calculate_G0(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk]


def Calculate_Gk(N_e, p_0, T_0, p_pp, T_pp, p_k, T_pv):
    # Потери:
    d_p0 = 0.05
    d_p_pp = 0.1
    d_p = 0.03
    # Параметры свежего пара
    point_0 = IAPWS97(P=p_0 * 10 ** (-6), T=T_0)
    s_0 = point_0.s
    h_0 = point_0.h
    v_0 = point_0.v
    #
    p_0_ = p_0 - 0.05 * p_0
    point_p_0_ = IAPWS97(P=p_0_ * 10 ** (-6), h=h_0)
    t_0_ = point_p_0_.T - 273.15
    s_0_ = point_p_0_.s
    v_0_ = point_p_0_.v
    # Теоретический процесс расширения в ЦВД
    p_1t = p_pp + 0.1 * p_pp
    point_1t = IAPWS97(P=p_1t * 10 ** (-6), s=s_0)
    t_1t = point_1t.T - 273.15
    h_1t = point_1t.h
    v_1t = point_1t.v
    #
    point_pp = IAPWS97(P=p_pp * 10 ** (-6), T=T_pp)
    h_pp = point_pp.h
    s_pp = point_pp.s
    v_pp = point_pp.v
    # Действительный процесс расширения в ЦВД
    H_0 = h_0 - h_1t
    eta_oi = 0.85
    H_i_cvd = H_0 * eta_oi
    h_1 = h_0 - H_i_cvd
    point_1 = IAPWS97(P=p_1t * 10 ** (-6), h=h_1)
    s_1 = point_1.s
    T_1 = point_1.T
    v_1 = point_1.v
    #
    p_pp_ = p_pp - 0.03 * p_pp
    point_pp_ = IAPWS97(P=p_pp_ * 10 ** (-6), h=h_pp)
    s_pp_ = point_pp_.s
    v_pp_ = point_pp_.v
    #
    point_kt = IAPWS97(P=p_k * 10 ** (-6), s=s_pp)
    T_kt = point_kt.T
    h_kt = point_kt.h
    v_kt = point_kt.v
    s_kt = s_pp
    #
    H_0_csdcnd = h_pp - h_kt
    eta_oi = 0.85
    H_i_csdcnd = H_0_csdcnd * eta_oi
    h_k = h_pp - H_i_csdcnd
    point_k = IAPWS97(P=p_k * 10 ** (-6), h=h_k)
    T_k = point_k.T
    s_k = point_k.s
    v_k = point_k.v
    #
    point_k_v = IAPWS97(P=p_k * 10 ** (-6), x=0)
    h_k_v = point_k_v.h
    s_k_v = point_k_v.s
    eta_oiI = (h_1 - h_0) / (h_1t - h_0)
    p_pv = 1.4 * p_0
    point_pv = IAPWS97(P=p_pv * 10 ** (-6), T=T_pv)
    h_pv = point_pv.h
    s_pv = point_pv.s
    #
    ksi_pp_oo = 1 - (1 - (T_k * (s_pp - s_k_v)) / ((h_0 - h_1t) + (h_pp - h_k_v))) / (
                1 - (T_k * (s_pp - s_pv)) / ((h_0 - h_1t) + (h_pp - h_pv)))
    T_0_ = IAPWS97(P=p_pv * 10 ** (-6), x=0).T
    T_ = (point_pv.T - point_k.T) / (T_0_ - point_k.T)
    if T_ <= 0.636364:
        ksi1 = -1.53 * T_ ** 2 + 2.1894 * T_ + 0.0048
    elif 0.636364 < T_ <= 0.736364:
        ksi1 = -1.3855 * T_ ** 2 + 2.00774 * T_ + 0.0321
    elif 0.736364 < T_ <= 0.863636:
        ksi1 = -2.6536 * T_ ** 2 + 4.2556 * T_ - 0.8569

    if T_ <= 0.631818:
        ksi2 = -1.7131 * T_ ** 2 + 2.3617 * T_ - 0.0142
    elif 0.631818 < T_ <= 0.718182:
        ksi2 = -2.5821 * T_ ** 2 + 3.689 * T_ - 0.4825
    elif 0.718182 < T_ <= 0.827273:
        ksi2 = -2.5821 * T_ ** 2 + 3.138 * T_ - 0.3626
    ksi = (ksi1 + ksi2) / 2
    ksi_r_pp = ksi * ksi_pp_oo
    eta_ir = (H_i_cvd + H_i_csdcnd) / (H_i_cvd + (h_pp - h_k_v)) * 1 / (1 - ksi_r_pp)
    H_i = eta_ir * ((h_0 - h_pv) + (h_pp - h_1))
    eta_m = 0.994
    eta_eg = 0.99
    G_0 = N_e / (H_i * eta_m * eta_eg * (10 ** 3))
    G_k = N_e / ((h_k - h_k_v) * eta_m * eta_eg * (10 ** 3)) * (1 / eta_ir - 1)
    return G_k


Gk = [Calculate_Gk(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk]

itog = pd.DataFrame({
    "Давление в конденсаторе": (list(range(2000, 10000, 500))),
    "КПД": [Calculate_G0_Gk(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk],
    "G_0": [Calculate_G0(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk],
    "G_k": [Calculate_Gk(N_e=Ne, p_0=p0, T_0=T0, p_pp=ppp, T_pp=Tpp, p_k=p, T_pv=Tpv) for p in pk]
})

x = (list(range(2000, 10000, 500)))
y = (eta)

p = figure(
    title='Зависимость КПД от давления в конденсаторе',
    x_axis_label='давление в конденсаторе',
    y_axis_label='КПД')

p.line(x, y, legend_label='Зависимость КПД от давления в конденсаторе', line_width=4)
st.bokeh_chart(p, use_container_width=True)

fighs = plt.figure()
point_0 = IAPWS97(P=p0 * 1e-6, T=T0)
p_0_d = p0 - delta_p_0
point_0_d = IAPWS97(P=p_0_d * 1e-6, h=point_0.h)
p_1t = ppp + delta_p_pp
point_1t = IAPWS97(P=p_1t * 10 ** (-6), s=point_0.s)
H_01 = point_0.h - point_1t.h
kpd_oi = 0.85
H_i_cvd = H_01 * kpd_oi
h_1 = point_0.h - H_i_cvd
point_1 = IAPWS97(P=p_1t * 1e-6, h=h_1)
point_pp = IAPWS97(P=ppp * 1e-6, T=Tpp)
p_pp_d = ppp - delta_p_pp
point_pp_d = IAPWS97(P=p_pp_d * 1e-6, h=point_pp.h)
point_kt = IAPWS97(P=pk_min * 1e-6, s=point_pp.s)
H_02 = point_pp.h - point_kt.h
kpd_oi = 0.85
H_i_csd_cnd = H_02 * kpd_oi
h_k = point_pp.h - H_i_csd_cnd
point_k = IAPWS97(P=pk_min * 1e-6, h=h_k)

s_0 = [point_0.s - 0.05, point_0.s, point_0.s + 0.05]
h_0 = [IAPWS97(P=p0 * 1e-6, s=s_).h for s_ in s_0]
s_1 = [point_0.s - 0.05, point_0.s, point_0.s + 0.18]
h_1 = [IAPWS97(P=p_1t * 1e-6, s=s_).h for s_ in s_1]
s_0_d = [point_0_d.s - 0.05, point_0_d.s, point_0_d.s + 0.05]
h_0_d = h_0
s_pp = [point_pp.s - 0.05, point_pp.s, point_pp.s + 0.05]
h_pp = [IAPWS97(P=ppp * 1e-6, s=s_).h for s_ in s_pp]
s_k = [point_pp.s - 0.05, point_pp.s, point_pp.s + 0.8]
h_k = [IAPWS97(P=pk_min * 1e-6, s=s_).h for s_ in s_k]
s_pp_d = [point_pp_d.s - 0.05, point_pp_d.s, point_pp_d.s + 0.05]
h_pp_d = h_pp

plt.plot([point_0.s, point_0.s, point_0_d.s, point_1.s], [point_1t.h, point_0.h, point_0.h, point_1.h], '-or')
plt.plot([point_pp.s, point_pp.s, point_pp_d.s, point_k.s], [point_kt.h, point_pp.h, point_pp.h, point_k.h], '-or')
plt.plot(s_0, h_0)
plt.plot(s_1, h_1)
plt.plot(s_0_d, h_0_d)
plt.plot(s_pp, h_pp)
plt.plot(s_k, h_k)
plt.plot(s_pp_d, h_pp_d)

for x, y, ind in zip([point_pp.s, point_k.s], [point_pp.h, point_k.h], ['{пп}', '{к}']):
    plt.text(x - 0.45, y + 40, '$h_' + ind + ' = %.2f $' % y)
for x, y, ind in zip([point_kt.s, point_pp_d.s], [point_kt.h, point_pp_d.h], ['{кт}', '{ппд}']):
    plt.text(x + 0.03, y + 40, '$h_' + ind + ' = %.2f $' % y)

for x, y, ind in zip([point_0.s, point_1.s], [point_0.h, point_1.h], ['{0}', '{1}']):
    plt.text(x - 0.01, y + 120, '$h_' + ind + ' = %.2f $' % y)

for x, y, ind in zip([point_1t.s, point_0_d.s], [point_1t.h, point_0_d.h], ['{1т}', '{0д}']):
    plt.text(x + 0.03, y - 60, '$h_' + ind + ' = %.2f $' % y)

plt.title("h - s диаграмма")
plt.xlabel("Значение энтропии s, кДж/(кг*С)")
plt.ylabel("Значение энтальпии h, кДж/кг")
plt.grid(True)
st.pyplot(fighs)

itog

st.write("Максимальный КПД:")
itog.iloc[0:1]


def iso_bar(wsp_point, min_s=-0.1, max_s=0.11, step_s=0.011, color='r'):
    if not isinstance(wsp_point, list):
        iso_bar_0_s = np.arange(wsp_point.s + min_s, wsp_point.s + max_s, step_s).tolist()
        iso_bar_0_h = [WSP(P=wsp_point.P, s=i).h for i in iso_bar_0_s]
    else:
        iso_bar_0_s = np.arange(wsp_point[0].s + min_s, wsp_point[1].s + max_s, step_s).tolist()
        iso_bar_0_h = [WSP(P=wsp_point[1].P, s=i).h for i in iso_bar_0_s]
    plt.plot(iso_bar_0_s, iso_bar_0_h, color)


st.write('# Курсовая работа Part II')

st.sidebar.header("Ввод параметров:")

st.sidebar.markdown('**1) Средний диаметр:**')
value_d = st.sidebar.slider('', 60, 150, (90, 111))
st.sidebar.write('Значения cредних диаметров: ', value_d[0], "-", value_d[1], "м")

st.sidebar.markdown('**2) Давление пара перед турбиной:**')
P_0 = st.sidebar.number_input('', value=29)
st.sidebar.write('P_0 = ', P_0, "МПа")

st.sidebar.markdown('**3) Температура пара перед турбиной**')
t_0 = st.sidebar.number_input('', value=575)
st.sidebar.write('t_0 = ', t_0, "°C")

st.sidebar.markdown('**4) Частота вращения ротора турбины**')
n_ = st.sidebar.number_input('', value=50)
st.sidebar.write('n = ', n_, "c^(-1)")

st.sidebar.markdown('**5) Расход водяного пара**')
G = st.sidebar.number_input('', value=448.34)
st.sidebar.write('G_0 = ', G, "кг/с")

st.sidebar.markdown('**6) Располагаемый теплоперепад ступени**')
H = st.sidebar.number_input('', value=110)
st.sidebar.write('H_0 = ', H, "кДж/кг")


def iso_bar(wsp_point, min_s=-0.1, max_s=0.11, step_s=0.011, color='b'):
    if not isinstance(wsp_point, list):
        iso_bar_0_s = np.arange(wsp_point.s + min_s, wsp_point.s + max_s, step_s).tolist()
        iso_bar_0_h = [IAPWS97(P=wsp_point.P, s=i).h for i in iso_bar_0_s]
    else:
        iso_bar_0_s = np.arange(wsp_point[0].s + min_s, wsp_point[1].s + max_s, step_s).tolist()
        iso_bar_0_h = [IAPWS97(P=wsp_point[1].P, s=i).h for i in iso_bar_0_s]
    plt.plot(iso_bar_0_s, iso_bar_0_h, color)


G_0 = 631.73 #кг/с
d = 1.08 #m
n = 60 #Гц
p_0 = 24.4 #МПа
T_0 = 552 + 273.15 #К
ro = 0.05 #степень реактивности
H_0 = 90 #кДж/кг
b_1 = 0.06 #м
b_2 = 0.03 #м
l_1 = 0.015 #м
alpha_1e = 12 #град
delta = 0.003 #перекрыша
kappa_vs = 0 #коэффициент использования выходной скорости
def callculate_optimum(d, p_0, T_0, n, G_0, H_0, ro, l_1, alpha_1e, b_1, delta, b_2, kappa_vs):
    u = M.pi*d*n
    point_0 = WSP(P = p_0, T = T_0)
    H_0_s = H_0*(1-ro)
    H_0_r = H_0*ro
    h_1t = point_0.h - H_0_s
    point_1t = WSP(h = h_1t, s = point_0.s)
    c_1t = (2000*H_0_s)**0.5
    M_1t = c_1t/point_1t.w
    mu_1 = 0.982 - 0.005*(b_1/l_1)
    F_1 = G_0*point_1t.v/mu_1/c_1t
    el_1 = F_1/M.pi/d/M.sin(M.radians(alpha_1e))
    e_opt=5*el_1**0.5
    if e_opt > 0.85:
        e_opt = 0.85
    l_1 = el_1/e_opt
    phi_s = 0.98 - 0.008*(b_1/l_1)
    c_1 = c_1t*phi_s
    alpha_1 = M.degrees(M.asin(mu_1/phi_s*M.sin(M.radians(alpha_1e))))
    w_1 = (c_1**2+u**2-2*c_1*u*M.cos(M.radians(alpha_1)))**0.5
    betta_1 = M.degrees(M.atan(M.sin(M.radians(alpha_1))/(M.cos(M.radians(alpha_1))-u/c_1)))
    delta_H_s = c_1t**2/2*(1-phi_s**2)
    h_1 = h_1t + delta_H_s*1e-3
    point_1 = WSP(P = point_1t.P, h = h_1)
    h_2t = h_1 - H_0_r
    point_2t = WSP(h = h_2t, s = point_1.s)
    w_2t = (2*H_0_r*1e3+w_1**2)**0.5
    l_2 = l_1 + delta
    mu_2 = 0.965-0.01*(b_2/l_2)
    M_2t = w_2t/point_2t.w
    F_2 = G_0*point_2t.v/mu_2/w_2t
    betta_2 = M.degrees(M.asin(F_2/(e_opt*M.pi*d*l_2)))
    point_1w = WSP(h = point_1.h+w_1**2/2*1e-3, s = point_1.s)
    psi_r = 0.96 - 0.014*(b_2/l_2)
    w_2 = psi_r*w_2t
    c_2 = (w_2**2+u**2-2*u*w_2*M.cos(M.radians(betta_2)))**0.5
    alpha_2 = M.degrees(M.atan(M.sin(M.radians(betta_2))/(M.cos(M.radians(betta_2))-u/w_2)))
    if alpha_2<0:
        alpha_2 = 180 + alpha_2
    delta_H_r = w_2t**2/2*(1-psi_r**2)
    h_2 = h_2t+delta_H_r*1e-3
    point_2 = WSP(P = point_2t.P, h = h_2)
    delta_H_vs = c_2**2/2
    E_0 = H_0 - kappa_vs*delta_H_vs
    etta_ol1 = (E_0*1e3 - delta_H_s-delta_H_r-(1-kappa_vs)*delta_H_vs)/(E_0*1e3)
    etta_ol2 = (u*(c_1*M.cos(M.radians(alpha_1))+c_2*M.cos(M.radians(alpha_2))))/(E_0*1e3)
    print(point_2.P,h_2)
    return etta_ol2, alpha_2


H_0 = [i for i in list(range(90, 111))]

eta = []
ucf = []
for i in H_0:
    ucf_1 = M.pi * d * n / (2000 * i) ** 0.5
    ucf.append(ucf_1)
    eta_ol, alpha = callculate_optimum(d, p_0, T_0, n, G_0, i, ro, l_1, alpha_1e, b_1, delta, b_2, kappa_vs)
    print(i, eta_ol, alpha, ucf_1)
    eta.append(eta_ol)

plt.plot(ucf, eta)
plt.show()


def iso_bar(wsp_point, min_s=-0.1, max_s=0.11, step_s=0.011, color='r'):
    if not isinstance(wsp_point, list):
        iso_bar_0_s = np.arange(wsp_point.s + min_s, wsp_point.s + max_s, step_s).tolist()
        iso_bar_0_h = [WSP(P=wsp_point.P, s=i).h for i in iso_bar_0_s]
    else:
        iso_bar_0_s = np.arange(wsp_point[0].s + min_s, wsp_point[1].s + max_s, step_s).tolist()
        iso_bar_0_h = [WSP(P=wsp_point[1].P, s=i).h for i in iso_bar_0_s]
    plt.plot(iso_bar_0_s, iso_bar_0_h, color)


def calc_optimal_H_0(rng):
    best_idx = -1
    max_value = -1
    for (H_0, index) in zip(rng, range(0, len(rng), 1)):
        print("Для значения H_0: ", H_0)
        u = M.pi * d * n
        print(f'u = {u:.2f} м/с')
        point_0 = WSP(P=p_0, T=T_0)
        print(f'h_0 = {point_0.h:.2f} кДж/кг')
        print(f's_0 = {point_0.s:.4f} кДж/(кг*К)')
        H_0s = H_0 * (1 - rho)
        H_0r = H_0 * rho
        h_1t = point_0.h - H_0s
        print(f'h_1т = {h_1t:.2f} кДж/кг')
        point_1t = WSP(h=h_1t, s=point_0.s)
        c_1t = (2000 * H_0s) ** 0.5
        print(f'c_1т = {c_1t:.2f} м/с')
        M_1t = c_1t / point_1t.w
        print(f'M_1т = {M_1t:.2f}')
        mu_1 = 0.982 - 0.005 * (b_1 / l_1)
        F_1 = G_0 * point_1t.v / mu_1 / c_1t
        k_tr = 0.0007
        Kappa_VS = 0
        u = M.pi * d * n
        c_f = M.sqrt(2000 * H_0)
        ucf = u / c_f
        xi_tr = k_tr * d ** 2 / F_1 * ucf ** 3
        print("xi_tr = ", xi_tr)
        if xi_tr > max_value:
            max_value = xi_tr
            best_idx = index
        print("\n\n\n\n\n")

    return rng[best_idx]


d = 1.08  # m
p_0 = 24.4  # МПа
T_0 = 552 + 273.15  # K
n = 60  # Гц
G_0 = 631.73  # кг/с
rho = 0.05
l_1 = 0.015  # м
alpha_1 = 12  # град
b_1 = 0.06  # м
Delta = 0.003  # м
b_2 = 0.03  # м
kappa_vs = 0
H_0 = calc_optimal_H_0(range(90, 111, 1))  # кДж/кг
print("Оптимальное значение H_0 ", H_0)

betta_1 = M.degrees(M.atan(M.sin(M.radians(alpha_1))/(M.cos(M.radians(alpha_1))-u/c_1)))
print(f'betta_1 = {betta_1:.2f} град')
delta_H_s = c_1t**2/2*(1-phi_s**2)
h_1 = h_1t + delta_H_s*1e-3
point_1 = WSP(P=point_1t.P, h=h_1)
h_2t = h_1 - H_0_r
point_2t = WSP(h=h_2t, s=point_1.s)
w_2t = (2000*H_0_r + w_1**2)**0.5
l_2 = l_1 + delta
mu_2 = 0.965 - 0.01*(b_2/l_2)
M_2t = w_2t/point_2t.w
F_2 = (G_0*point_2t.v)/(mu_2*w_2t)
betta_2e = M.degrees(M.asin(F_2/(e_opt*M.pi*d*l_2)))
print(f'betta_2e = {betta_2e:.4f} град')

point_1w = WSP(h = point_1.h+w_1**2/2*1e-3, s = point_1.s)
def plot_hs_stage_t(x_lim,y_lim):
    plot_hs_nozzle_t(x_lim,y_lim)
    plt.plot([point_0.s,point_1.s],[point_0.h,point_1.h],'bo-')
    plt.plot([point_1.s,point_2t.s],[point_1.h,point_2t.h], 'ro-')
    plt.plot([point_1.s,point_1.s],[point_1w.h, point_1.h],'ro-')
    iso_bar(point_2t,-0.02,0.02,0.001,'y')
    iso_bar(point_1w,-0.005,0.005,0.001,'c')
plot_hs_stage_t([6.19,6.26],[3230,3380])
plt.show()
c_1u = c_1*M.cos(M.radians(alpha_1))
c_1a = c_1*M.sin(M.radians(alpha_1))
w_1u = c_1u - u
w_2a = w_2*M.sin(M.radians(betta_2e))
w_2u = w_2*M.cos(M.radians(betta_2e))
c_2u=w_2u-u
print(c_1u,w_1u)
w_1_tr = [0, 0, -w_1u, -c_1a]
c_1_tr = [0, 0, -c_1u, -c_1a]
u_1_tr = [-w_1u, -c_1a, -u, 0]

w_2_tr = [0, 0, w_2u, -w_2a]
c_2_tr = [0, 0, c_2u, -w_2a]
u_2_tr = [w_2u,-w_2a, -u, 0]
ax = plt.axes()
ax.arrow(*c_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='r', ec='r')
ax.arrow(*w_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='b', ec='b')
ax.arrow(*u_1_tr, head_width=5, length_includes_head = True,head_length=20, fc='g', ec='g')
ax.arrow(*c_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='r', ec='r')
ax.arrow(*w_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='b', ec='b')
ax.arrow(*u_2_tr, head_width=5, length_includes_head = True,head_length=20, fc='g', ec='g')
plt.text(-2*c_1u/3, -3*c_1a/4, '$c_1$', fontsize=20)
plt.text(-2*w_1u/3, -3*c_1a/4, '$w_1$', fontsize=20)
plt.text(2*c_2u/3, -3*w_2a/4, '$c_2$', fontsize=20)
plt.text(2*w_2u/3, -3*w_2a/4, '$w_2$', fontsize=20)
plt.show()

H_i = E_0 - delta_H_r*1e-3 - delta_H_s*1e-3 - (1-Kappa_VS)*delta_H_vs*1e-3 - Delta_Hub - Delta_Htr - Delta_H_parc
print("""Использованный теплоперепад ступени  
           H_i = %.3f кДж/кг""" % H_i)
eta_oi = H_i/E_0
print("""Внутренний относительный КПД ступени  
        eta_oi  = %.3f""" % eta_oi)
N_i = G_0*H_i
print("""Внутреняя мощность ступени  
            N_i = = %.2f кВт""" % N_i)

