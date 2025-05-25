import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon
import warnings

EPSILON = 1e-10

def is_categorical_or_object_dtype(series):
    """
    Проверяет, является ли признак категориальным.

    Args:
        series (pd.Series): Pandas Series с данными по одному признаку.
                                 Должна быть числовой или категориальной/object типа.
    Returns:
        bool: True, если признак категориальный и False, если некатегориальный
    """
    return isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == 'object'

def calculate_psi(base_series, target_series, feature_name, n_bins=10):
    """
    Рассчитывает Population Stability Index (PSI) для числового или категориального признака.

    Args:
        base_series (pd.Series): Pandas Series с данными для базовой (обучающей) выборки по одному признаку.
                                 Должна быть числовой или категориальной/object типа.
        target_series (pd.Series): Pandas Series с данными для целевой (текущей) выборки по тому же признаку.
                                  Должна быть того же типа, что и base_series.
        feature_name (str): Название признака (используется для сообщений об ошибках/предупреждений).
        n_bins (int, optional): Количество интервалов (бинов) для числовых признаков.
                                Defaults to 10. Игнорируется для категориальных признаков.

    Returns:
        dict: Словарь с результатами PSI: {'metric_type': 'psi', 'metric_value': значение_psi, 'note': заметки}.
              Возвращает NaN в случае ошибки или неподдерживаемого типа данных.
    """
    try:
        if pd.api.types.is_numeric_dtype(base_series) and pd.api.types.is_numeric_dtype(target_series):
            # Числовой признак: биннинг по квантилям базовой выборки
            combined_series = pd.concat([base_series.dropna(), target_series.dropna()])
            if combined_series.nunique() < n_bins:
                 # Если уникальных значений меньше, чем бинов, используем уникальные значения
                 bins = np.unique(combined_series)
                 if len(bins) < 2: # Если после удаления NaN осталось только одно значение или 0
                     return {'metric_type': 'psi', 'metric_value': 0.0, 'note': 'Меньше 2 уникальных значений'}
            else:
                 try:
                     # Пытаемся использовать квантили базовой выборки
                     _, bins = pd.qcut(base_series.dropna(), q=n_bins, retbins=True, duplicates='drop')

                 except Exception:
                     # Если квантили посчитать не удалось (например, много одинаковых значений), используем равные интервалы по общему диапазону
                     warnings.warn(f"Не удалось рассчитать квантили для признака '{feature_name}'. Используем интервалы равной ширины.")
                     bin_min, bin_max = combined_series.min(), combined_series.max()
                     if bin_min == bin_max:
                          return {'metric_type': 'psi', 'metric_value': 0.0, 'note': 'Признак с постоянным значением'}
                     bins = np.linspace(bin_min, bin_max, n_bins + 1)

            bins[0] = bins[0] - EPSILON
            bins[-1] = bins[-1] + EPSILON

            # Применяем биннинг к обеим выборкам, считаем частоты
            base_binned = pd.cut(base_series, bins=bins, right=True, include_lowest=True).value_counts().sort_index()
            target_binned = pd.cut(target_series, bins=bins, right=True, include_lowest=True).value_counts().sort_index()

            # Рассчитываем доли в бинах
            # Убеждаемся, что сумма частот > 0 перед делением
            base_sum = base_binned.sum()
            target_sum = target_binned.sum()
            if base_sum == 0 or target_sum == 0:
                 warnings.warn(f"Нулевое общее количество в базовой или целевой выборке для признака '{feature_name}'. Невозможно рассчитать PSI.")
                 return {'metric_type': 'psi', 'metric_value': np.nan, 'note': 'Нулевое общее количество в базовой или целевой выборке'}


            base_props = base_binned / base_sum
            target_props = target_binned / target_sum

            # Переиндексируем, чтобы убедиться, что бины совпадают и заполнить 0 там, где бин пуст
            all_bins = base_props.index.union(target_props.index)
            base_props = base_props.reindex(all_bins, fill_value=0)
            target_props = target_props.reindex(all_bins, fill_value=0)

            # Расчет PSI
            # Формула: sum((p_target - p_base) * log(p_target / p_base))
            psi_values = (target_props - base_props) * np.log((target_props + EPSILON) / (base_props + EPSILON))
            psi = psi_values.sum()

            return {'metric_type': 'psi', 'metric_value': psi}

        elif (is_categorical_or_object_dtype(base_series) and
              is_categorical_or_object_dtype(target_series)):
            # Категориальный признак: каждая категория - бин
            base_counts = base_series.value_counts()
            target_counts = target_series.value_counts()

            # Объединяем все уникальные категории из обеих выборок
            all_categories = base_counts.index.union(target_counts.index)

            # Рассчитываем доли в категориях, заполняя 0 для отсутствующих
            # Убеждаемся, что сумма частот > 0 перед делением
            base_sum = base_counts.sum()
            target_sum = target_counts.sum()
            if base_sum == 0 or target_sum == 0:
                 warnings.warn(f"Нулевое общее количество в базовой или целевой выборке для признака '{feature_name}'. Невозможно рассчитать PSI.")
                 return {'metric_type': 'psi', 'metric_value': np.nan, 'note': 'Нулевое общее количество в базовой или целевой выборке'}


            base_props = base_counts.reindex(all_categories, fill_value=0) / base_sum
            target_props = target_counts.reindex(all_categories, fill_value=0) / target_sum

            psi_values = (target_props - base_props) * np.log((target_props + EPSILON) / (base_props + EPSILON))
            psi = psi_values.sum()

            return {'metric_type': 'psi', 'metric_value': psi}

        else:
            warnings.warn(f"Неподдерживаемый тип данных для PSI на признаке '{feature_name}': {base_series.dtype}")
            return {'metric_type': 'psi', 'metric_value': np.nan, 'note': 'Неподдерживаемый тип'}

    except Exception as e:
        warnings.warn(f"Ошибка при расчете PSI для признака '{feature_name}': {e}")
        return {'metric_type': 'psi', 'metric_value': np.nan, 'note': f'Ошибка: {e}'}

def calculate_ks_test(base_series, target_series, feature_name):
    """
    Рассчитывает критерий Колмогорова-Смирнова для двух выборок (числовые признаки).

    Args:
        base_series (pd.Series): Pandas Series с числовыми данными для базовой (обучающей) выборки.
                                 Должна содержать только числовые значения.
        target_series (pd.Series): Pandas Series с числовыми данными для целевой (текущей) выборки по тому же признаку.
                                   Должна содержать только числовые значения.
        feature_name (str): Название признака (используется для сообщений об ошибках/предупреждений).

    Returns:
        dict or None: Словарь с результатами K-S теста ('ks_stat', 'p_value') или None,
                      если тест неприменим (например, нечисловые данные, слишком мало наблюдений).
    """
    # Проверка, что данные являются числовыми
    if not pd.api.types.is_numeric_dtype(base_series) or not pd.api.types.is_numeric_dtype(target_series):
        warnings.warn(f"Тест Колмогорова-Смирнова требует числовых данных. Пропускаем признак '{feature_name}'.")
        return None

    # KS тест работает только на числовых данных без NaN
    base_clean = base_series.dropna()
    target_clean = target_series.dropna()

    # KS тест требует минимум 2 не-NaN значения в каждой выборке
    if len(base_clean) < 2 or len(target_clean) < 2:
         warnings.warn(f"Тест Колмогорова-Смирнова требует минимум 2 не-NaN значения в обеих выборках. Пропускаем признак '{feature_name}'.")
         return None

    try:
        ks_statistic, p_value = ks_2samp(base_clean, target_clean)
        return {'metric_type': 'ks_stat', 'metric_value': ks_statistic, 'p_value': p_value}
    except Exception as e:
        warnings.warn(f"Ошибка при расчете теста Колмогорова-Смирнова для признака '{feature_name}': {e}")
        return {'metric_type': 'ks_stat', 'metric_value': np.nan, 'p_value': np.nan, 'note': f'Ошибка: {e}'}

def calculate_chi2_test(base_series, target_series, feature_name):
    """
    Рассчитывает критерий Хи-квадрат для двух выборок (категориальные признаки).

    Args:
        base_series (pd.Series): Pandas Series с категориальными/object данными для базовой выборки.
                                 Должна содержать только категориальные или строковые значения.
        target_series (pd.Series): Pandas Series с категориальными/object данными для целевой выборки по тому же признаку.
                                   Должна содержать только категориальные или строковые значения.
        feature_name (str): Название признака (используется для сообщений об ошибках/предупреждений).

    Returns:
        dict or None: Словарь с результатами Хи-квадрат теста ('chi2_stat', 'p_value') или None,
                      если тест неприменим (например, не категориальные данные, нулевые суммы по строкам/столбцам).
    """
    # Проверка, что данные подходят для Хи-квадрат теста (категориальные или object)
    if not (is_categorical_or_object_dtype(base_series) and
            is_categorical_or_object_dtype(target_series)):
        warnings.warn(f"Критерий Хи-квадрат требует категориальных или строковых данных. Пропускаем признак '{feature_name}'.")
        return None

    # Объединяем все уникальные категории из обеих выборок, удаляя NaN
    all_categories = pd.concat([base_series.dropna(), target_series.dropna()]).unique()

    if len(all_categories) < 2:
         warnings.warn(f"Критерий Хи-квадрат требует минимум 2 категории. Пропускаем признак '{feature_name}'.")
         return None

    # Строим таблицы частот, включая категории, отсутствующие в одной из выборок (с 0)
    base_counts = base_series.value_counts().reindex(all_categories, fill_value=0)
    target_counts = target_series.value_counts().reindex(all_categories, fill_value=0)

    # Строим таблицу сопряженности (DataFrame с частотами)
    contingency_table = pd.DataFrame({'Base': base_counts, 'Target': target_counts})

    # Хи-квадрат тест требует, чтобы сумма по строке или столбцу не была нулевой,
    # а также общее количество наблюдений > 0.
    if contingency_table.sum().sum() == 0 or any(contingency_table.sum(axis=1) == 0):
         warnings.warn(f"Критерий Хи-квадрат требует ненулевых сумм по строкам/столбцам. Пропускаем признак '{feature_name}'.")
         return None

    try:
        # Выполнение теста Хи-квадрат
        # chi2_contingency возвращает: statistic, p-value, degrees of freedom, expected frequencies
        chi2_statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        return {'metric_type': 'chi2_stat', 'metric_value': chi2_statistic, 'p_value': p_value}
    except ValueError as ve:
         warnings.warn(f"ValueError при расчете критерия Хи-квадрат для признака '{feature_name}': {ve}. Может произойти при очень малых количествах.")
         return {'metric_type': 'chi2_stat', 'metric_value': np.nan, 'p_value': np.nan, 'note': f'Ошибка ValueError: {ve}'}
    except Exception as e:
        warnings.warn(f"Ошибка при расчете критерия Хи-квадрат для признака '{feature_name}': {e}")
        return {'metric_type': 'chi2_stat', 'metric_value': np.nan, 'p_value': np.nan, 'note': f'Ошибка: {e}'}


def calculate_jsd(base_series, target_series, feature_name, n_bins=10):
    """
    Рассчитывает Jensen-Shannon Distance (квадратный корень из JSD) для двух выборок.
    Возвращает метрику расстояния, которая лежит в диапазоне [0, 1].
    Для числовых признаков используется биннинг по квантилям базовой выборки.
    Для категориальных признаков каждая категория - отдельный "бин".

    Args:
        base_series (pd.Series): Pandas Series с данными для базовой (обучающей) выборки по одному признаку.
                                 Должна быть числовой или категориальной/object типа.
        target_series (pd.Series): Pandas Series с данными для целевой (текущей) выборки по тому же признаку.
                                  Должна быть того же типа, что и base_series.
        feature_name (str): Название признака (используется для сообщений об ошибках/предупреждений).
        n_bins (int, optional): Количество интервалов (бинов) для числовых признаков.
                                Defaults to 10. Игнорируется для категориальных признаков.

    Returns:
        dict: Словарь с результатами JSD: {'metric_type': 'js_distance', 'metric_value': значение_js_distance, 'note': заметки}.
              Возвращает NaN в случае ошибки или неподдерживаемого типа данных.
    """
    try:
        if pd.api.types.is_numeric_dtype(base_series) and pd.api.types.is_numeric_dtype(target_series):
            # Числовой признак: биннинг по квантилям базовой выборки (те же бины, что и для PSI)
            combined_series = pd.concat([base_series.dropna(), target_series.dropna()])
            if combined_series.nunique() < n_bins:
                 bins = np.unique(combined_series)
                 if len(bins) < 2:
                      return {'metric_type': 'js_distance', 'metric_value': 0.0, 'note': 'Меньше 2 уникальных значений'}
            else:
                 try:
                     # Получаем границы бинов из qcut по базовой выборке
                     _, bins = pd.qcut(base_series.dropna(), q=n_bins, retbins=True, duplicates='drop')
                 except Exception:
                      warnings.warn(f"Не удалось рассчитать квантили для JSD на признаке '{feature_name}'. Используем интервалы равной ширины.")
                      bin_min, bin_max = combined_series.min(), combined_series.max()
                      if bin_min == bin_max:
                           return {'metric_type': 'js_distance', 'metric_value': 0.0, 'note': 'Признак с постоянным значением'}
                      bins = np.linspace(bin_min, bin_max, n_bins + 1)

            # Добавляем небольшое смещение к границам, чтобы включить крайние значения
            bins[0] = bins[0] - EPSILON
            bins[-1] = bins[-1] + EPSILON

            # Применяем биннинг к обеим выборкам, считаем частоты
            base_binned = pd.cut(base_series, bins=bins, right=True, include_lowest=True).value_counts().sort_index()
            target_binned = pd.cut(target_series, bins=bins, right=True, include_lowest=True).value_counts().sort_index()

            # Рассчитываем доли в бинах
            # Убеждаемся, что сумма частот > 0 перед делением
            base_sum = base_binned.sum()
            target_sum = target_binned.sum()
            if base_sum == 0 or target_sum == 0:
                 warnings.warn(f"Нулевое общее количество в базовой или целевой выборке для признака '{feature_name}'. Невозможно рассчитать JSD.")
                 return {'metric_type': 'js_distance', 'metric_value': np.nan, 'note': 'Нулевое общее количество в базовой или целевой выборке'}

            base_props = base_binned / base_sum
            target_props = target_binned / target_sum


            # Переиндексируем, чтобы убедиться, что бины совпадают и заполнить 0 там, где бин пуст
            all_bins = base_props.index.union(target_props.index)
            base_props = base_props.reindex(all_bins, fill_value=0)
            target_props = target_props.reindex(all_bins, fill_value=0)


            # Используем scipy функцию jensenshannon, она возвращает JS Distance (sqrt(JSD))
            # Добавляем EPSILON перед передачей в scipy для устойчивости к нулям в массивах
            js_distance = jensenshannon(base_props.values + EPSILON, target_props.values + EPSILON)

            return {'metric_type': 'js_distance', 'metric_value': js_distance}

        elif (is_categorical_or_object_dtype(base_series) and
              is_categorical_or_object_dtype(target_series)):
            # Категориальный признак: каждая категория - бин
            base_counts = base_series.value_counts()
            target_counts = target_series.value_counts()

            # Объединяем все уникальные категории
            all_categories = base_counts.index.union(target_counts.index)

            # Рассчитываем доли в категориях, заполняя 0 для отсутствующих
            # Убеждаемся, что сумма частот > 0 перед делением
            base_sum = base_counts.sum()
            target_sum = target_counts.sum()
            if base_sum == 0 or target_sum == 0:
                 warnings.warn(f"Нулевое общее количество в базовой или целевой выборке для признака '{feature_name}'. Невозможно рассчитать JSD.")
                 return {'metric_type': 'js_distance', 'metric_value': np.nan, 'note': 'Нулевое общее количество в базовой или целевой выборке'}

            base_props = base_counts.reindex(all_categories, fill_value=0) / base_sum
            target_props = target_counts.reindex(all_categories, fill_value=0) / target_sum

            # Используем scipy функцию jensenshannon, она возвращает JS Distance (sqrt(JSD))
            # Добавляем EPSILON перед передачей в scipy для устойчивости к нулям в массивах
            js_distance = jensenshannon(base_props.values + EPSILON, target_props.values + EPSILON)

            return {'metric_type': 'js_distance', 'metric_value': js_distance}

        else:
            warnings.warn(f"Неподдерживаемый тип данных для JSD на признаке '{feature_name}': {base_series.dtype}")
            return {'metric_type': 'js_distance', 'metric_value': np.nan, 'note': 'Неподдерживаемый тип'}

    except Exception as e:
        warnings.warn(f"Ошибка при расчете JSD для признака '{feature_name}': {e}")
        return {'metric_type': 'js_distance', 'metric_value': np.nan, 'note': f'Ошибка: {e}'}


def check_data_drift(df_base, df_target):
    """
    Проверяет дрейф данных для всех подходящих признаков в DataFrame,
    автоматически определяя их тип (числовой или нечисловой).

    Args:
        df_base (pd.DataFrame): pandas DataFrame, представляющий обучающую (базовую) выборку.
        df_target (pd.DataFrame): pandas DataFrame, представляющий текущую (целевую) выборку.

    Returns:
        pd.DataFrame: DataFrame с результатами проверки дрейфа для каждого признака и каждой метрики.
                      Содержит колонки: 'feature_name', 'metric_type', 'metric_value', 'p_value' (если применимо), 'note'.
                      Без колонки 'check_timestamp'.
    """
    drift_results = []
    numerical_features = []
    categorical_features = []
    skipped_features = []

    # 1. Автоматическое определение типов признаков на основе базовой выборки
    print("Автоматическое определение типов объектов на основе базового датафрейма...")
    for feature in df_base.columns:
        # Проверяем, присутствует ли признак в обеих выборках
        if feature not in df_target.columns:
            warnings.warn(f"Признак '{feature}' найден в базовых данных, но не в целевых. Пропускаем этот признак.")
            skipped_features.append(feature)
            continue  # Пропускаем признаки, отсутствующие в целевой выборке

        base_dtype = df_base[feature].dtype
        target_dtype = df_target[feature].dtype

        # Проверяем совместимость типов и относим к числовым или нечисловым
        if pd.api.types.is_numeric_dtype(base_dtype) and pd.api.types.is_numeric_dtype(target_dtype):
            numerical_features.append(feature)
        elif (is_categorical_or_object_dtype(df_base[feature]) and
              is_categorical_or_object_dtype(df_target[feature])):
            # Дополнительно преобразуем нечисловые столбцы в category для единообразия,
            # если они еще не такие.
            if not isinstance(df_base[feature].dtype, pd.CategoricalDtype):
                df_base[feature] = df_base[feature].astype('category')
            if not isinstance(df_target[feature].dtype, pd.CategoricalDtype):
                df_target[feature] = df_target[feature].astype('category')
            categorical_features.append(feature)
        else:
            warnings.warn(
                f"Признак '{feature}' имеет неподдерживаемые или несовпадающие типы данных (Базовый: {base_dtype}, Целевой: {target_dtype}). Пропускаем.")
            skipped_features.append(feature)

    print(f"Обнаружены числовые признаки: {numerical_features}")
    print(f"Обнаружены категориальные признаки: {categorical_features}")
    if skipped_features:
        print(f"Пропущенные признаки из-за отсутствия или неподдерживаемых/несовпадающих типов: {skipped_features}")

    # 2. Проверка дрейфа для определенных числовых признаков
    print("\nПроверка дрейфа для числовых признаков...")
    for feature in numerical_features:
        print(f"  - Признак: {feature}")

        # Рассчет PSI для числового признака
        psi_result = calculate_psi(df_base[feature], df_target[feature], feature, n_bins=10)
        psi_result['feature_name'] = feature
        drift_results.append(psi_result)

        # Рассчет KS Test для числового признака
        ks_result = calculate_ks_test(df_base[feature], df_target[feature], feature)
        if ks_result:
            ks_result['feature_name'] = feature
            drift_results.append(ks_result)

        # Рассчет JSD для числового признака
        jsd_result = calculate_jsd(df_base[feature], df_target[feature], feature, n_bins=10)
        jsd_result['feature_name'] = feature
        drift_results.append(jsd_result)

    # 3. Проверка дрейфа для определенных категориальных признаков
    print("\nПроверка дрейфа для категориальных признаков...")
    for feature in categorical_features:
        print(f"  - Признак: {feature}")

        # Рассчет PSI для категориального признака
        psi_result = calculate_psi(df_base[feature], df_target[feature], feature)  # n_bins не используется
        psi_result['feature_name'] = feature
        drift_results.append(psi_result)

        # Рассчет Chi-squared Test для категориального признака
        chi2_result = calculate_chi2_test(df_base[feature], df_target[feature], feature)
        if chi2_result:
            chi2_result['feature_name'] = feature
            drift_results.append(chi2_result)

        # Рассчет JSD для категориального признака
        jsd_result = calculate_jsd(df_base[feature], df_target[feature], feature)  # n_bins не используется
        jsd_result['feature_name'] = feature
        drift_results.append(jsd_result)

    # 4. Преобразование списка словарей в DataFrame для удобства
    return pd.DataFrame(drift_results)

if __name__ == "__main__":
    base_file_path = 'data_p1.csv'
    target_file_path = 'data_p2.csv'

    try:
        print(f"Загрузка базовых данных из {base_file_path}...")
        df_base = pd.read_csv(base_file_path)
        print("Базовые данные загружены успешно.")

        print(f"Загрузка целевых данных из {target_file_path}...")
        df_target = pd.read_csv(target_file_path)
        print("Целевые данные загружены успешно.")

        print("Размер базовых данных:", df_base.shape)
        print("Размер целевых данных:", df_target.shape)

    except FileNotFoundError as e:
        print(f"Ошибка: Файл не найден - {e}. Пожалуйста, проверьте пути к файлам.")
        exit()
    except Exception as e:
        print(f"Произошла ошибка при загрузке данных из файлов: {e}")
        exit()

    target_column_name = 'cb_person_default_on_file'

    df_base_features = df_base.drop(columns=[target_column_name], errors='ignore')
    df_target_features = df_target.drop(columns=[target_column_name], errors='ignore')

    drift_results_df = check_data_drift(df_base_features, df_target_features)

    print("\n--- Результаты проверки дрейфа данных ---")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(drift_results_df)

    output_csv_path = 'data_drift_results.csv'
    try:
        drift_results_df.to_csv(output_csv_path, index=False)
        print(f"\nРезультаты сохранены в {output_csv_path}")
    except Exception as e:
        print(f"\nОшибка сохранения результатов в CSV файл '{output_csv_path}': {e}")
