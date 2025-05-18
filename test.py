import pandas as pd
import random
from datetime import datetime, timedelta


def add_timestamps_to_csv(input_csv, output_csv):
    """
    Добавляет столбец с таймстемпами в прошлое к существующему CSV файлу.

    Аргументы:
        input_csv (str): Путь к входному CSV файлу.
        output_csv (str): Путь для сохранения выходного CSV файла с новым столбцом.
    """
    try:
        # Чтение существующего CSV файла
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_csv}' не найден.")
        return
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {e}")
        return

    # Получение сегодняшней даты
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    all_timestamps = []
    current_date = today

    # Генерируем таймстемпы, пока не наберется достаточно для всех строк в DataFrame
    while len(all_timestamps) < len(df):
        # Определяем случайное количество строк для текущего дня (от 10 до 20)
        num_rows_for_day = random.randint(10, 20)

        # Генерируем случайные таймстемпы в течение текущего дня
        day_timestamps = [current_date + timedelta(seconds=random.randint(0, 24 * 3600 - 1)) for _ in
                          range(num_rows_for_day)]

        # Сортируем таймстемпы в пределах дня (опционально, но может быть полезно)
        day_timestamps.sort(reverse=True)  # Сортируем в обратном порядке, чтобы сохранить тренд "в прошлое"

        all_timestamps.extend(day_timestamps)

        # Переходим к предыдущему дню
        current_date -= timedelta(days=1)

    # Обрезаем список таймстемпов до количества строк в исходном DataFrame
    # Это нужно, если сгенерировано больше таймстемпов, чем строк
    all_timestamps = all_timestamps[:len(df)]

    # Добавляем новый столбец с таймстемпами в DataFrame
    df['timestamp'] = all_timestamps

    # Сохранение обновленного DataFrame в новый CSV файл
    try:
        df.to_csv(output_csv, index=False)
        print(f"Файл успешно обновлен и сохранен как '{output_csv}'")
    except Exception as e:
        print(f"Произошла ошибка при сохранении файла: {e}")


# --- Пример использования ---
if __name__ == "__main__":
    input_csv_filename = 'data_p2_predicted.csv'  # Замените на имя вашего входного файла
    output_csv_filename = 'data_p2_predicted_with_timestamps.csv'  # Имя выходного файла

    # Вызов функции для добавления таймстемпов
    add_timestamps_to_csv(input_csv_filename, output_csv_filename)