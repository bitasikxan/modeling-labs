import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from prettytable import PrettyTable
import math
matplotlib.use('TkAgg')

print("=== ЗАПУСК МОДЕЛЮВАННЯ ===")

class Customer:
    def __init__(self, arrival_step):
        self.denied = np.random.rand() < 0.1
        self.goods = np.random.geometric(p=0.3)
        self.service_time = self._calculate_service_time()
        self.arrival_step = arrival_step

    def _calculate_service_time(self):
        time = (self.goods * 3) + 1
        if self.denied:
            time += 4
        return time


# --- ІМІТАЦІЙНА МОДЕЛЬ ---
def run_simulation(num_employees, num_of_steps=40, n_runs=100):
    results = {
        'lost_percent': [],
        'idle_prob': [],
        'avg_queue': [],
        'avg_wait': []
    }

    for _ in range(n_runs):
        step = 1
        queue = deque()
        lost_clients = 0
        total_arrived = 0
        workers = [0] * num_employees
        total_idle_time = 0

        # Накопичувач для "людино-хвилин" у черзі
        queue_person_minutes = 0

        while step <= num_of_steps:
            # 1. Генерація (Нестаціонарний потік)
            if 1 <= step <= 16:
                lambda_value = 4
            elif 17 <= step <= 28:
                lambda_value = 6
            else:
                lambda_value = 8

            arrived_count = np.random.poisson(lam=lambda_value)
            total_arrived += arrived_count

            for _ in range(arrived_count):
                if len(queue) >= 15:
                    lost_clients += 1
                else:
                    new_cust = Customer(arrival_step=step)
                    queue.append(new_cust)

            # 2. Обробка часу (похвилинний крок для точності підрахунку часу)
            time_budget = 15
            while time_budget > 0:
                # Хто вільний - бере роботу
                free_workers_indices = [i for i, w in enumerate(workers) if w <= 0]
                for idx in free_workers_indices:
                    if len(queue) > 0:
                        cust = queue.popleft()
                        workers[idx] = cust.service_time

                # Статистика простою
                all_free = all(w <= 0 for w in workers)
                if all_free and len(queue) == 0:
                    step_delta = time_budget
                    total_idle_time += step_delta
                    time_budget = 0
                else:
                    # Шукаємо найближчу подію
                    busy_times = [w for w in workers if w > 0]
                    step_delta = time_budget
                    if busy_times:
                        step_delta = min(time_budget, min(busy_times))

                    time_budget -= step_delta

                    # Накопичуємо час очікування
                    # Якщо 5 людей чекали 3 хвилини, це 15 людино-хвилин очікування
                    queue_person_minutes += len(queue) * step_delta

                    for i in range(num_employees):
                        if workers[i] > 0:
                            workers[i] -= step_delta
            step += 1

        # Підсумки прогону
        sim_time_minutes = num_of_steps * 15
        entered_clients = total_arrived - lost_clients  # Ті, хто реально потрапив у систему

        loss_pct = (lost_clients / total_arrived * 100) if total_arrived > 0 else 0
        idle_prob = total_idle_time / sim_time_minutes
        avg_q_len = queue_person_minutes / sim_time_minutes

        # Середній час очікування = (Людино-хвилини) / (Кількість людей, що зайшли)
        avg_wait_val = (queue_person_minutes / entered_clients) if entered_clients > 0 else 0

        results['lost_percent'].append(loss_pct)
        results['idle_prob'].append(idle_prob)
        results['avg_queue'].append(avg_q_len)
        results['avg_wait'].append(avg_wait_val)

    return {
        'lost_percent': np.mean(results['lost_percent']),
        'idle_prob': np.mean(results['idle_prob']),
        'avg_queue': np.mean(results['avg_queue']),
        'avg_wait': np.mean(results['avg_wait'])
    }


# --- АНАЛІТИЧНА МОДЕЛЬ (M/M/c/K) ---
def calculate_analytical(c, m=15):
    # Середні параметри (обчислено в звіті)
    lambda_hour = 23.2
    mu_hour = 5.26

    r = lambda_hour / mu_hour
    rho = r / c

    # 1. P0
    sum_part = sum([(r ** k) / math.factorial(k) for k in range(c)])
    if abs(rho - 1.0) < 1e-6:
        term2 = (r ** c / math.factorial(c)) * (m + 1)
    else:
        term2 = (r ** c / math.factorial(c)) * ((1 - rho ** (m + 1)) / (1 - rho))
    p0 = 1 / (sum_part + term2)

    # 2. P_loss
    K_sys = c + m
    p_loss = (r ** K_sys / (math.factorial(c) * (c ** (K_sys - c)))) * p0

    # 3. Lq (Середня черга)
    if abs(rho - 1.0) < 1e-6:
        Lq = (r ** c / math.factorial(c)) * (m * (m + 1) / 2) * p0
    else:
        numerator = 1 - rho ** m - m * (rho ** m) * (1 - rho)
        denominator = (1 - rho) ** 2
        Lq = (r ** c / math.factorial(c)) * rho * (numerator / denominator) * p0


    lambda_eff = lambda_hour * (1 - p_loss)

    if lambda_eff > 0:
        Wq_hours = Lq / lambda_eff
        Wq_min = Wq_hours * 60
    else:
        Wq_min = 0

    return rho, p0, p_loss * 100, Lq, Wq_min


# --- ЕКСПЕРИМЕНТ ---
print("Розрахунок для c = 1...7")

employee_counts = [1, 2, 3, 4, 5, 6, 7]
sim_data = {'P_loss': [], 'P_idle': [], 'L_q': [], 'W_q': []}
ana_data = {'rho': [],'P_loss': [], 'P_idle': [], 'L_q': [], 'W_q': []}

for c in employee_counts:
    # Імітація
    s_res = run_simulation(c, n_runs=100)
    sim_data['P_loss'].append(s_res['lost_percent'])
    sim_data['P_idle'].append(s_res['idle_prob'])
    sim_data['L_q'].append(s_res['avg_queue'])
    sim_data['W_q'].append(s_res['avg_wait'])

    # Аналітика
    rho, a_p0, a_ploss, a_lq, a_wq = calculate_analytical(c)
    ana_data['rho'].append(rho)
    ana_data['P_loss'].append(a_ploss)
    ana_data['P_idle'].append(a_p0)
    ana_data['L_q'].append(a_lq)
    ana_data['W_q'].append(a_wq)

# --- ТАБЛИЦЯ ---
print("\nСередні значення для обох моделей:")
table = PrettyTable()
table.field_names = ["Працівники", "Втрати(%) IM / AM", "Простій IM / AM", "Черга IM / AM", "Час очік.(хв) IM / AM", "Стабільність (лише АМ)"]


for i, c in enumerate(employee_counts):
    mark = ""
    if sim_data['P_loss'][i] < 5.0 and sim_data['P_idle'][i] < 0.15:
        mark = "(*)"

    table.add_row([
        f"{c} {mark}",
        f"{sim_data['P_loss'][i]:.1f} / {ana_data['P_loss'][i]:.1f}",
        f"{sim_data['P_idle'][i]:.3f} / {ana_data['P_idle'][i]:.3f}",
        f"{sim_data['L_q'][i]:.1f} / {ana_data['L_q'][i]:.1f}",
        f"{sim_data['W_q'][i]:.1f} / {ana_data['W_q'][i]:.1f}",
        f"{ana_data['rho'][i]:.2f}"
    ])

print(table)
print("* - к-сть працівників, яка задовольняє умову")

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Аналіз ефективності пункту видачі замовлень", fontsize=16)

# 1. Втрати
axs[0, 0].plot(employee_counts, sim_data['P_loss'], 'y-o', label='Імітація')
axs[0, 0].plot(employee_counts, ana_data['P_loss'], 'g--x', label='Аналітика')
axs[0, 0].set_title("Втрати клієнтів (%)")
axs[0, 0].axhline(y=5, color='gray', linestyle=':', label='Ліміт 5%')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# 2. Простій
axs[0, 1].plot(employee_counts, sim_data['P_idle'], 'y-o', label='Імітація')
axs[0, 1].plot(employee_counts, ana_data['P_idle'], 'g--x', label='Аналітика')
axs[0, 1].set_title("Ймовірність простою (P0)")
axs[0, 1].axhline(y=0.15, color='k', linestyle=':', label='Ліміт 0.15')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# 3. Довжина черги
axs[1, 0].plot(employee_counts, sim_data['L_q'], 'y-o', label='Імітація')
axs[1, 0].plot(employee_counts, ana_data['L_q'], 'g--x', label='Аналітика')
axs[1, 0].set_title("Середня довжина черги (людей)")
axs[1, 0].axhline(y=15, color='k', linestyle='--', label='Max (15)')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# 4. Час очікування
axs[1, 1].plot(employee_counts, sim_data['W_q'], 'y-o', label='Імітація')
axs[1, 1].plot(employee_counts, ana_data['W_q'], 'g--x', label='Аналітика')
axs[1, 1].set_title("Середній час очікування (хв)")
axs[1, 1].set_xlabel("Кількість працівників")
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()