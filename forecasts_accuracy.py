# script_performance_monitoring.py
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, confusion_matrix,
    precision_score, recall_score, accuracy_score, f1_score,
    brier_score_loss
)
import warnings
import datetime
import math
import os # Для проверки существования файла

# Игнорируем предупреждения, которые могут возникать из sklearn, если данных недостаточно
# (например, zero_division в precision_score, если нет предсказанных положительных)
# или из pandas при работе с NaT/NaN.
# Можно настроить более тонко, если это необходимо.
# warnings.filterwarnings('ignore') # Закомментировано, чтобы видеть предупреждения о недостатке данных


EPSILON = 1e-10

# Хелпер для расчета KS статистики производительности
def calculate_ks_performance(y_true, y_score):
    """
    Рассчитывает KS статистику для оценки производительности модели.
    """
    # Создаем DataFrame только с необходимыми колонками и удаляем строки с NaN
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score}).dropna()

    if df.empty:
         #warnings.warn("calculate_ks_performance: Input data is empty after dropping NaN.")
         warnings.warn("calculate_ks_performance: Входные данные пусты после удаления NaN.")
         return np.nan

    df = df.sort_values(by='y_score')

    if df['y_true'].nunique() < 2:
        # warnings.warn("KS performance requires both 0 and 1 classes to be present.")
        warnings.warn("Расчет KS производительности требует наличия классов 0 и 1.")
        return np.nan

    actual_0 = df[df['y_true'] == 0]['y_score']
    actual_1 = df[df['y_true'] == 1]['y_score']

    if len(actual_0) == 0 or len(actual_1) == 0:
         # warnings.warn("KS performance requires non-empty samples for both classes.")
         warnings.warn("Расчет KS производительности требует непустых выборок для обоих классов.")
         return np.nan

    n0 = len(actual_0)
    n1 = len(actual_1)

    thresholds = np.unique(df['y_score'])

    # Использование searchsorted для эффективности
    cdf0 = np.searchsorted(np.sort(actual_0), thresholds, side='right') / n0
    cdf1 = np.searchsorted(np.sort(actual_1), thresholds, side='right') / n1

    ks_stat = np.max(np.abs(cdf0 - cdf1))

    return ks_stat

# --- Основная функция мониторинга производительности ---

def calculate_performance_metrics(df_prod, score_col, actual_col, timestamp_col, evaluation_frequency='M', threshold=0.5):
    """
    Рассчитывает метрики производительности модели на данных из продакшена
    за определенные временные периоды и возвращает результаты в широком формате
    (каждая метрика в своей колонке).

    Args:
        df_prod (pd.DataFrame): DataFrame с данными из продакшена
                                (должен содержать скор, фактический исход, время).
        score_col (str): Название колонки с предсказанными вероятностями (скорами) [0, 1].
        actual_col (str): Название колонки с фактическим исходом (0 или 1).
        timestamp_col (str): Название колонки с меткой времени предсказания.
                              Должна содержать данные, которые pandas может преобразовать в datetime.
        evaluation_frequency (str, optional): Периодичность оценки ('overall', 'D', 'W', 'M', 'Q', 'Y').
                                              Defaults to 'M'.
        threshold (float, optional): Порог классификации для метрик, зависящих от порога.
                                     Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame с результатами мониторинга производительности по периодам.
                      Колонки: 'evaluation_period', а также колонки для каждой метрики.
    """
    # Список для хранения результатов для каждого периода (в виде словарей)
    period_results_list = []

    # 1. Валидация и подготовка данных
    if not all(col in df_prod.columns for col in [score_col, actual_col, timestamp_col]):
        missing_cols = [col for col in [score_col, actual_col, timestamp_col] if col not in df_prod.columns]
        raise ValueError(f"DataFrame должен содержать колонки: '{score_col}', '{actual_col}', '{timestamp_col}'. Отсутствуют: {missing_cols}")

    df = df_prod.copy()

    # Преобразование колонки времени в формат datetime
    try:
        # Используем errors='coerce' для преобразования ошибочных значений в NaT
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        # Удаляем строки, где не удалось преобразовать время (станут NaT)
        initial_rows = len(df)
        df.dropna(subset=[timestamp_col], inplace=True)
        if len(df) < initial_rows:
            warnings.warn(f"Удалено {initial_rows - len(df)} строк из входных данных из-за неверного формата метки времени в колонке '{timestamp_col}'.")

        if df.empty and len(df_prod) > 0:
            raise ValueError(f"Колонка метки времени '{timestamp_col}' не может быть преобразована в формат даты/времени ни для одной строки, или все строки были удалены из-за неверного формата.")
        elif df.empty:
            print("Входной DataFrame пуст после обработки меток времени. Нет метрик производительности для расчета.")
            return pd.DataFrame() # Возвращаем пустой DataFrame если нет данных

    except Exception as e:
         # Это исключение будет перехвачено, если pd.to_datetime вернет ошибку напрямую
        raise ValueError(f"Не удалось преобразовать колонку метки времени '{timestamp_col}' в формат даты/времени: {e}")


    # Проверка значений целевой переменной (только 0 и 1, пропуски допустимы)
    # Проверяем только среди НЕ-NaN значений
    if not df[actual_col].dropna().isin([0, 1]).all():
         warnings.warn(f"Колонка фактического исхода '{actual_col}' должна идеально содержать только значения 0 и 1 (NaN допустимы). Найдены другие значения среди не-NaN записей.")

    # Проверка значений скора (в диапазоне [0, 1], пропуски допустимы)
    # Проверяем только среди НЕ-NaN значений
    if not (df[score_col].dropna() >= 0).all() or not (df[score_col].dropna() <= 1).all():
         warnings.warn(f"Колонка со скором '{score_col}' должна содержать значения от 0 до 1 (вероятности) среди не-NaN записей. Найдены значения вне этого диапазона.")


    # 2. Определение временных периодов для оценки
    if evaluation_frequency == 'overall':
        periods = [('Overall', df)]
    else:
        if df.empty:
             periods = []
        else:
            df = df.sort_values(by=timestamp_col) # Сортируем по времени для корректной группировки периодов
            # Создаем колонку с началом периода
            # .dt.to_period(freq) может вернуть ошибку на пустых или проблемных данных,
            # оборачиваем в try
            try:
                df['period_start'] = df[timestamp_col].dt.to_period(evaluation_frequency).dt.start_time
                # Группируем по периодам
                periods = [(str(period_start.date()), group_df) for period_start, group_df in df.groupby('period_start')]
            except Exception as e:
                 raise ValueError(f"Ошибка группировки данных по частоте '{evaluation_frequency}' в колонке меток времени '{timestamp_col}': {e}")


    # 3. Расчет метрик для каждого периода
    print(f"Расчет метрик производительности для {len(periods)} периодов...")

    for period_name, period_df in periods:
        # Словарь для хранения метрик для текущего периода
        current_period_metrics = {'evaluation_period': period_name}

        if period_df.empty:
            print(f"  - Период {period_name}: Нет данных, пропускаем.")
            current_period_metrics['Note'] = 'Нет данных'
            period_results_list.append(current_period_metrics)
            continue

        # Удаляем строки, где actual_col или score_col являются NaN для расчета метрик
        # Большинство метрик sklearn не обрабатывают NaN в y_true или y_score
        # Колонки actual_col и score_col используются для большинства метрик
        period_df_clean_metrics = period_df.dropna(subset=[actual_col, score_col])

        y_true = period_df_clean_metrics[actual_col]
        y_score = period_df_clean_metrics[score_col]

        # Проверяем, является ли y_true числовым и приводим к int, если необходимо
        if pd.api.types.is_numeric_dtype(y_true):
             y_true = y_true.astype(int)


        # Проверка, достаточно ли данных для расчета метрик (минимум 2 наблюдения и оба класса для большинства)
        # LogLoss, BrierScore, AUC/Gini, KS_Performance требуют разное минимальное количество данных
        # Будем добавлять метрики по мере возможности, а не скипать весь период из-за одной метрики.
        min_samples_required = 2 # Минимальное количество записей для большинства метрик
        two_classes_required = (y_true.nunique() >= 2) # Требуется 2 класса для большинства метрик

        if len(y_true) < min_samples_required:
            warnings.warn(f"  - Период {period_name}: Недостаточно данных ({len(y_true)} строк после очистки для расчета метрик). Пропускаем большинство метрик.")
            current_period_metrics['Note'] = f'Недостаточно данных ({len(y_true)} строк)'
            # Добавляем NaN для всех ожидаемых метрик, чтобы колонки были консистентными
            # (Список метрик определен ниже)
            for metric_name_base in ['AUC', 'Gini', 'LogLoss', 'BrierScore', 'KS_Performance', 'Accuracy', 'Precision', 'Recall', 'F1']:
                 if metric_name_base in ['Accuracy', 'Precision', 'Recall', 'F1']:
                      current_period_metrics[f'{metric_name_base}_@{threshold}'] = np.nan
                 else:
                      current_period_metrics[metric_name_base] = np.nan

            period_results_list.append(current_period_metrics)
            continue

        if not two_classes_required:
             warnings.warn(f"  - Период {period_name}: Только один класс ({y_true.nunique()} уникальных классов после очистки). Пропускаем метрики, требующие оба класса (AUC, Gini, LogLoss, BrierScore, KS_Performance, Precision, Recall, F1). Accuracy может быть рассчитана, если данных > 0.")
             current_period_metrics['Note'] = f'Один класс ({y_true.nunique()} уникальных)'
             # Добавляем NaN для метрик, требующих 2 класса
             for metric_name_base in ['AUC', 'Gini', 'LogLoss', 'BrierScore', 'KS_Performance', 'Precision', 'Recall', 'F1']:
                  if metric_name_base in ['Precision', 'Recall', 'F1']:
                      current_period_metrics[f'{metric_name_base}_@{threshold}'] = np.nan
                  else:
                       current_period_metrics[metric_name_base] = np.nan
             # Accuracy может быть посчитана, если len(y_true) >= 1, но ее смысл при одном классе ограничен
             # В sklearn accuracy_score требует y_true, y_pred длины >= 1.
             # Порог 0.5 применим, даже если один класс.
             # Посчитаем Accuracy отдельно, если данных достаточно, даже если один класс.

        # Если данных достаточно (>= 2)
        print(f"  - Период: {period_name} ({len(y_true)} записей для расчета)")

        # --- Расчет метрик ---
        # Расчет метрик, не зависящих от порога (AUC, Gini, LogLoss, Brier, KS) - требуют 2 класса
        if two_classes_required:
            try:
                auc = roc_auc_score(y_true, y_score)
                current_period_metrics['AUC'] = auc
                current_period_metrics['Gini'] = 2 * auc - 1
            except Exception as e:
                 warnings.warn(f"    Ошибка при расчете AUC/Gini для {period_name}: {e}")
                 current_period_metrics['AUC'] = np.nan
                 current_period_metrics['Gini'] = np.nan

            try:
                 # LogLoss требует, чтобы y_score был массивом вероятностей для классов
                 # Для бинарной классификации: [1-p, p]
                 # Убеждаемся, что y_score находится строго в (0, 1) для log
                 y_score_clipped = np.clip(y_score, EPSILON, 1 - EPSILON) # Клиппируем для устойчивости
                 y_score_proba = np.vstack([1 - y_score_clipped, y_score_clipped]).T
                 logloss = log_loss(y_true, y_score_proba)
                 current_period_metrics['LogLoss'] = logloss
            except Exception as e:
                 warnings.warn(f"    Ошибка при расчете LogLoss для {period_name}: {e}")
                 current_period_metrics['LogLoss'] = np.nan

            try:
                 brier = brier_score_loss(y_true, y_score)
                 current_period_metrics['BrierScore'] = brier
            except Exception as e:
                 warnings.warn(f"    Ошибка при расчете Brier Score для {period_name}: {e}")
                 current_period_metrics['BrierScore'] = np.nan

            try:
                 # Расчет KS статистики производительности
                 ks_perf = calculate_ks_performance(y_true, y_score)
                 current_period_metrics['KS_Performance'] = ks_perf
                 if np.isnan(ks_perf): # calculate_ks_performance уже выдает предупреждение
                      pass
            except Exception as e:
                warnings.warn(f"    Ошибка при расчете KS Performance для {period_name}: {e}")
                current_period_metrics['KS_Performance'] = np.nan
        else:
             # Если 2 класса не присутствуют, добавляем NaN для этих метрик
             current_period_metrics['AUC'] = np.nan
             current_period_metrics['Gini'] = np.nan
             current_period_metrics['LogLoss'] = np.nan
             current_period_metrics['BrierScore'] = np.nan
             current_period_metrics['KS_Performance'] = np.nan


        # Расчет метрик, зависящих от порога (Accuracy, Precision, Recall, F1)
        # Требуют y_true и y_pred
        # Accuracy требует только >=1 sample
        # Precision/Recall/F1 требуют наличия положительного класса (1) в y_true И предсказанных положительных для Precision
        try:
            y_pred = (y_score >= threshold).astype(int)

            # Accuracy
            if len(y_true) >= 1: # Accuracy можно посчитать даже с одним классом, если есть хотя бы 1 запись
                acc = accuracy_score(y_true, y_pred)
                current_period_metrics[f'Accuracy_@{threshold}'] = acc
            else:
                 current_period_metrics[f'Accuracy_@{threshold}'] = np.nan


            # Precision, Recall, F1 требуют наличия положительного класса (1) в y_true
            if 1 in y_true.unique():
               # zero_division=0 устанавливает метрику в 0, если нет предсказанных положительных (Precision)
               # или нет фактических положительных (Recall, F1)
               prec = precision_score(y_true, y_pred, zero_division=0)
               rec = recall_score(y_true, y_pred, zero_division=0)
               f1 = f1_score(y_true, y_pred, zero_division=0)

               current_period_metrics[f'Precision_@{threshold}'] = prec
               current_period_metrics[f'Recall_@{threshold}'] = rec
               current_period_metrics[f'F1_@{threshold}'] = f1
            else:
               warnings.warn(f"    Период {period_name}: Пропускаем Precision/Recall/F1, так как нет положительного класса (1) в фактических значениях после очистки.")
               current_period_metrics[f'Precision_@{threshold}'] = np.nan
               current_period_metrics[f'Recall_@{threshold}'] = np.nan
               current_period_metrics[f'F1_@{threshold}'] = np.nan


        except Exception as e:
             warnings.warn(f"    Ошибка при расчете метрик, зависящих от порога, для {period_name}: {e}")
             current_period_metrics[f'Accuracy_@{threshold}'] = np.nan
             current_period_metrics[f'Precision_@{threshold}'] = np.nan
             current_period_metrics[f'Recall_@{threshold}'] = np.nan
             current_period_metrics[f'F1_@{threshold}'] = np.nan

        # Добавляем словарь с метриками для текущего периода в список
        period_results_list.append(current_period_metrics)


    # 4. Преобразование списка словарей в DataFrame
    results_df = pd.DataFrame(period_results_list)

    # 5. Финальная обработка DataFrame
    if not results_df.empty:
         # Преобразуем 'evaluation_period' в datetime для DataLens, если период не 'Overall'
         if evaluation_frequency != 'overall':
            # Убеждаемся, что колонка существует, прежде чем преобразовывать
            if 'evaluation_period' in results_df.columns:
                results_df['evaluation_period'] = pd.to_datetime(results_df['evaluation_period'])
         else: # Для 'Overall' преобразуем в фиктивную дату начала для сортировки
            if 'evaluation_period' in results_df.columns:
                # Находим строки с 'Overall'
                overall_rows_mask = results_df['evaluation_period'] == 'Overall'
                # Присваиваем фиктивную дату только этим строкам
                results_df.loc[overall_rows_mask, 'evaluation_period'] = pd.to_datetime('1900-01-01')
                # Для всех остальных строк (если были какие-то другие строковые значения, хотя не должны бы)
                # пытаемся преобразовать в datetime
                other_rows_mask = ~overall_rows_mask
                results_df.loc[other_rows_mask, 'evaluation_period'] = pd.to_datetime(results_df.loc[other_rows_mask, 'evaluation_period'], errors='coerce')


         # Определяем полный список метрик, которые должны быть в колонках
         expected_metric_cols = [
             'AUC', 'Gini', 'LogLoss', 'BrierScore', 'KS_Performance',
             f'Accuracy_@{threshold}', f'Precision_@{threshold}', f'Recall_@{threshold}', f'F1_@{threshold}'
         ]
         # Добавляем колонку Note, если она присутствует в данных хоть раз
         if any('Note' in d for d in period_results_list):
              expected_metric_cols.append('Note')


         # Убеждаемся, что все ожидаемые колонки присутствуют, добавляя их с NaN, если отсутствуют
         # Это полезно, если какой-то период вообще не позволил посчитать определенную метрику
         for col in expected_metric_cols:
             if col not in results_df.columns:
                 results_df[col] = np.nan

         # Переупорядочиваем колонки: 'evaluation_period' первая, затем метрики, затем Note
         ordered_cols = ['evaluation_period'] + [col for col in expected_metric_cols if col != 'Note']
         if 'Note' in results_df.columns:
              ordered_cols.append('Note')

         # Фильтруем ordered_cols, чтобы оставить только те, которые реально есть в results_df
         ordered_cols_present = [col for col in ordered_cols if col in results_df.columns]
         results_df = results_df[ordered_cols_present]


    return results_df


# --- Блок запуска скрипта мониторинга производительности (чтение из файла) ---
if __name__ == '__main__':
    print("--- Запуск скрипта мониторинга производительности (Чтение из файла) ---")

    # --- НАСТРОЙКА ---
    # Укажите путь к вашему CSV файлу с данными из продакшена
    # Пример файла можно создать вручную или использовать сгенерированный ранее
    # Структура файла должна содержать как минимум 3 колонки: timestamp, actual_value (0/1), score (0..1)
    # Пример данных в файле:
    # timestamp,score_col,actual_col
    # 2023-01-15 10:30:00,0.85,1
    # 2023-01-20 11:00:00,0.12,0
    # 2023-02-10 09:15:00,0.67,1
    # ...
    prod_filepath = 'data_p2_predicted_with_timestamps.csv' # <-- УКАЖИТЕ ВАШ ПУТЬ К ФАЙЛУ
    # Укажите названия колонок в вашем файле
    score_column = 'probability_default' # <-- УКАЖИТЕ НАЗВАНИЕ КОЛОНКИ СО СКОРОМ (вероятность [0, 1])
    actual_column = 'cb_person_default_on_file' # <-- УКАЖИТЕ НАЗВАНИЕ КОЛОНКИ С ФАКТИЧЕСКИМ ИСХОДОМ (0 или 1)
    timestamp_column = 'timestamp' # <-- УКАЖИТЕ НАЗВАНИЕ КОЛОНКИ С МЕТКОЙ ВРЕМЕНИ (формат даты/времени)

    # Параметры оценки
    eval_freq = 'M' # Периодичность оценки: 'overall' (вся выборка), 'D' (день), 'W' (неделя), 'M' (месяц), 'Q' (квартал), 'Y' (год)
    prediction_threshold = 0.5 # Порог для Accuracy, Precision, Recall, F1


    # --- ПРОВЕРКА СУЩЕСТВОВАНИЯ ФАЙЛА ---
    if not os.path.exists(prod_filepath):
        raise FileNotFoundError(f"Файл данных из продакшена не найден: {prod_filepath}")

    # --- ЗАГРУЗКА ДАННЫХ ---
    try:
        # При чтении CSV может понадобиться указать разделитель (sep=',' или ';') или кодировку (encoding='utf-8' и т.д.)
        # Учитывая, что это CSV, разделитель чаще всего запятая.
        print(f"Загрузка данных из {prod_filepath}...")
        df_prod_data = pd.read_csv(prod_filepath)
        print(f"Загружено {len(df_prod_data)} строк из {prod_filepath}")
    except Exception as e:
        raise IOError(f"Ошибка при чтении CSV файла продакшена: {e}")

    # --- ВЫПОЛНЕНИЕ МОНИТОРИНГА ПРОИЗВОДИТЕЛЬНОСТИ ---
    try:
        print("\nВыполнение расчета метрик производительности...")
        performance_results = calculate_performance_metrics(
            df_prod_data,
            score_column,
            actual_column,
            timestamp_column,
            evaluation_frequency=eval_freq,
            threshold=prediction_threshold
        )

        print("\n--- Результаты мониторинга производительности модели ---")
        print(performance_results)

        # --- СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ---
        output_csv_path = 'performance_monitoring_results.csv' # Путь для сохранения результатов
        if not performance_results.empty:
            try:
                # Сохраняем с индексом False, чтобы не записывать индекс DataFrame в файл
                performance_results.to_csv(output_csv_path, index=False)
                print(f"\nРезультаты мониторинга производительности сохранены в {output_csv_path}")
            except Exception as e:
                print(f"\nОшибка при сохранении результатов мониторинга производительности в {output_csv_path}: {e}")
        else:
            print("\nНет результатов мониторинга производительности для сохранения (DataFrame пуст).")

    except Exception as e:
         print(f"\nПроизошла ошибка во время мониторинга производительности: {e}")