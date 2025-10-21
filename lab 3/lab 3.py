import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque
from prettytable import PrettyTable

matplotlib.use('TkAgg')


class Customer:
    def __init__(self, arrival_step):
        self.denied = np.random.rand() < 0.1
        self.goods = np.random.geometric(p=0.3)
        self.service_time = self._calculate_service_time()
        self.arrival_step = arrival_step
        self.queue_time = 0

    def _calculate_service_time(self):
        time = (self.goods * 3) + 1
        if self.denied:
            time += 4
        return time


def simulate_day(num_of_steps=40):

    step = 1
    queue = deque()
    lost_clients = 0
    processed_clients = []
    work_left_on_current_client = 0
    current_customer = None

    queue_len_history = []
    arrivals_per_step = []
    time_labels = []

    while step <= num_of_steps:
        if 1 <= step <= 16:
            lambda_value = 4
        elif 17 <= step <= 28:
            lambda_value = 6
        else:
            lambda_value = 8

        arrived_count = np.random.poisson(lam=lambda_value)
        arrivals_per_step.append(arrived_count)

        for _ in range(arrived_count):
            if len(queue) >= 15:
                lost_clients += 1
            else:
                new_cust = Customer(arrival_step=step)
                new_cust.queue_time = work_left_on_current_client + sum(c.service_time for c in queue)
                queue.append(new_cust)

        queue_len_history.append(len(queue))
        current_hour = 10 + (step - 1) * 15 // 60
        current_min = (step - 1) * 15 % 60
        time_labels.append(f"{current_hour:02}:{current_min:02}")

        time_budget = 15
        while time_budget > 0:
            if work_left_on_current_client == 0:
                if len(queue) > 0:
                    current_customer = queue.popleft()
                    work_left_on_current_client = current_customer.service_time
                else:
                    break
            time_to_work = min(time_budget, work_left_on_current_client)
            work_left_on_current_client -= time_to_work
            time_budget -= time_to_work
            if work_left_on_current_client == 0 and current_customer is not None:
                processed_clients.append(current_customer)
                current_customer = None

        step += 1

    df_customers = pd.DataFrame([{
        "arrival_step": c.arrival_step,
        "queue_time": c.queue_time,
        "goods": c.goods,
        "service_time": c.service_time,
        "denied": c.denied
    } for c in processed_clients])

    df_queue = pd.DataFrame({
        "step": range(1, num_of_steps + 1),
        "queue_len": queue_len_history,
        "arrivals": arrivals_per_step,
        "time": time_labels
    })

    summary = {
        "total_arrived": sum(arrivals_per_step),
        "served": len(processed_clients),
        "lost": lost_clients,
        "left_in_queue": len(queue),
        "avg_queue_len": np.mean(queue_len_history),
        "avg_goods": df_customers["goods"].mean() if not df_customers.empty else 0,
        "avg_service_time": df_customers["service_time"].mean() if not df_customers.empty else 0,
        "avg_wait_time": df_customers["queue_time"].mean() if not df_customers.empty else 0
    }

    return df_customers, df_queue, summary

# за 1 день
df_customers_day, df_queue_day, summary_day = simulate_day()

table_day = PrettyTable()
table_day.field_names = ["Показник", "Значення"]
table_day.add_row(["Загальна к-сть клієнтів", summary_day["total_arrived"]])
table_day.add_row(["К-сть обслужених клієнтів", summary_day["served"]])
table_day.add_row(["К-сть втрачених клієнтів", summary_day["lost"]])
table_day.add_row(["Середня довжина черги", round(summary_day["avg_queue_len"], 2)])
table_day.add_row(["Середня к-сть товару покупця", round(summary_day["avg_goods"], 2)])
table_day.add_row(["Середній час обслуговування (хв)", round(summary_day["avg_service_time"], 2)])
table_day.add_row(["Середній час очікування (хв)", round(summary_day["avg_wait_time"], 2)])

print("\n--- Зведена таблиця за 1 день ---")
print(table_day)

df_all_customers = pd.concat([simulate_day()[0] for _ in range(100)], ignore_index=True)

df_stats_100 = df_all_customers[["goods", "service_time", "queue_time"]].describe(percentiles=[0.25, 0.5, 0.75]).T
df_stats_100.rename(columns={
    "count": "Кількість",
    "mean": "Середнє",
    "std": "Ст. відхилення",
    "min": "Мінімум",
    "25%": "25% квартиль",
    "50%": "Медіана (50%)",
    "75%": "75% квартиль",
    "max": "Максимум"
}, inplace=True)

df_stats_100.index = df_stats_100.index.map({
    "goods": "Кількість товарів",
    "service_time": "Час обслуговування (хв)",
    "queue_time": "Час очікування (хв)"
})

# за 100 днів
table_100 = PrettyTable()
table_100.field_names = ["Показник"] + df_stats_100.index.tolist()

for col in df_stats_100.columns:
    row = [col] + list(df_stats_100[col].round(2).values)
    table_100.add_row(row)

print("\n--- Статистика за 100 днів ---")
print(table_100)

# графіки
if df_customers_day.empty:
    print("\nЖоден клієнт не був обслужений. Графіки неможливо побудувати.")
else:
    goods_list = df_customers_day["goods"].tolist()
    service_time_list = df_customers_day["service_time"].tolist()
    wait_time_list = df_customers_day["queue_time"].tolist()
    arrivals_per_step = df_queue_day["arrivals"].tolist()
    time_labels = df_queue_day["time"].tolist()
    queue_len_history = df_queue_day["queue_len"].tolist()

    arrivals_array = np.array(arrivals_per_step)
    hourly_arrivals, hours = [], []
    for i in range(10, 20):
        start_step = (i - 10) * 4
        end_step = (i - 9) * 4
        total_for_hour = arrivals_array[start_step:min(end_step, len(arrivals_array))].sum()
        hourly_arrivals.append(total_for_hour)
        hours.append(f"{i}:00-{i + 1}:00")

    fig, axs = plt.subplots(2, 3, figsize=(12, 12))  # зменшено
    fig.suptitle("Аналіз моделювання роботи пункту видачі", fontsize=18)

    axs[0, 0].plot(time_labels, queue_len_history, color='b', marker='.')
    axs[0, 0].set_title("1. Динаміка довжини черги протягом дня")
    axs[0, 0].set_xlabel("Час")
    axs[0, 0].set_ylabel("Клієнтів в черзі")
    axs[0, 0].set_xticks(time_labels[::4])
    axs[0, 0].grid(True)

    max_goods = max(goods_list)
    bins_goods = np.arange(0.5, max_goods + 1.6, 1)
    axs[0, 1].hist(goods_list, bins=bins_goods, edgecolor='black', alpha=0.8)
    axs[0, 1].set_title("2. Розподіл кількості товарів (в 1 замовленні)")
    axs[0, 1].set_xlabel("Кількість товарів")
    axs[0, 1].set_ylabel("Кількість клієнтів")
    axs[0, 1].set_xticks(range(1, max_goods + 2))
    axs[0, 1].grid(True, axis='y')

    axs[0, 2].hist(service_time_list, bins=20, edgecolor='black', color='green', alpha=0.8)
    axs[0, 2].set_title("3. Розподіл часу обслуговування")
    axs[0, 2].set_xlabel("Час обслуговування (хв)")
    axs[0, 2].set_ylabel("Кількість клієнтів")
    axs[0, 2].grid(True, axis='y')

    axs[1, 0].hist(wait_time_list, bins=20, edgecolor='black', color='red', alpha=0.8)
    axs[1, 0].set_title("4. Розподіл часу очікування в черзі")
    axs[1, 0].set_xlabel("Час очікування (хв)")
    axs[1, 0].set_ylabel("Кількість клієнтів")
    axs[1, 0].grid(True, axis='y')

    axs[1, 1].scatter(goods_list, service_time_list, alpha=0.6, edgecolors='black')
    axs[1, 1].set_title("5. Залежність часу обслуговування від к-сті товарів")
    axs[1, 1].set_xlabel("Кількість товарів")
    axs[1, 1].set_ylabel("Час обслуговування (хв)")
    axs[1, 1].grid(True)

    axs[1, 2].bar(hours, hourly_arrivals, color='purple', edgecolor='black', alpha=0.8)
    axs[1, 2].set_title("6. Розподіл прибуття покупців (за годинами)")
    axs[1, 2].set_xlabel("Година")
    axs[1, 2].set_ylabel("Кількість прибулих клієнтів")
    axs[1, 2].tick_params(axis='x', rotation=45)
    axs[1, 2].grid(True, axis='y')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
