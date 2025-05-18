import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --- Настройки: Укажите свои пути к файлам и списки признаков ---

# Путь к файлу с сохраненной моделью ИЛИ ПОЛНЫМ ПАЙПЛАЙНОМ (предпочтительный вариант)
# Если у вас сохранен весь пайплайн (препроцессор + модель), укажите путь к нему здесь.
# Если у вас сохранены только модель и препроцессор ОТДЕЛЬНО, укажите здесь путь только к модели.
model_file_path = 'logistic_regression_credit_scoring_model.joblib' # или 'credit_scoring_pipeline.joblib'

# Путь к файлу с сохраненным объектом предобработки (например, ColumnTransformer).
# ЭТОТ ФАЙЛ НУЖЕН ТОЛЬКО, ЕСЛИ В model_file_path СОХРАНЕНА ТОЛЬКО МОДЕЛЬ, А НЕ ВЕСЬ ПАЙПЛАЙН.
# Если у вас сохранен весь пайплайн, можете оставить этот путь пустым или None.
preprocessor_file_path = 'credit_scoring_preprocessor.joblib' # <--- УКАЖИТЕ ПУТЬ ИЛИ None

# Путь к вашему исходному CSV файлу с данными, для которых нужно сделать предсказания.
# Этот файл должен содержать НЕОБРАБОТАННЫЕ признаки, как до предобработки при обучении.
data_file_path = 'data_p2.csv' # <--- ЗАМЕНИТЕ НА ИМЯ ВАШЕГО ФАЙЛА ДАННЫХ

# Путь, по которому будет сохранен новый CSV файл с добавленными предсказаниями.
output_file_path = 'data_p2_predicted.csv' # <--- МОЖЕТЕ ИЗМЕНИТЬ ИМЯ ВЫХОДНОГО ФАЙЛА

# Списки имен числовых и категориальных признаков В ВАШИХ ИСХОДНЫХ ДАННЫХ (data_file_path)
# Эти списки ОЧЕНЬ ВАЖНЫ! Они должны ТОЧНО соответствовать признакам,
# на основе которых обучался ваш препроцессор.
# --- ЗАМЕНИТЕ ЭТИ СПИСКИ НА РЕАЛЬНЫЕ ИМЕНА ПРИЗНАКОВ ИЗ ВАШЕГО ИСХОДНОГО ФАЙЛА ---
numerical_features = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_status',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]
categorical_features = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade'
]
# ------------------------------------------------------------------

# Объединяем списки для проверки наличия всех необходимых исходных признаков
all_input_features = numerical_features + categorical_features
if not all_input_features:
    print("Ошибка: Списки numerical_features и categorical_features пусты.")
    print("Пожалуйста, укажите хотя бы один признак из вашего исходного CSV файла.")
    exit()

# --- 1. Загрузка модели И/ИЛИ Пайплайна ---
print("Загрузка модели/пайплайна...")
model_or_pipeline = None
preprocessor = None
is_pipeline = False

try:
    # Загружаем объект из пути к файлу модели/пайплайна
    loaded_object = joblib.load(model_file_path)
    print(f"Объект успешно загружен из: {model_file_path}")

    # Проверяем, является ли загруженный объект пайплайном scikit-learn
    if isinstance(loaded_object, Pipeline):
        print("   Загруженный объект является пайплайном.")
        is_pipeline = True
        model_or_pipeline = loaded_object # В этом случае, переменная model_or_pipeline - это весь пайплайн
    else:
        # Если загружен не пайплайн, считаем, что это только модель
        print("   Загруженный объект является только моделью.")
        model_or_pipeline = loaded_object

except FileNotFoundError:
    print(f"Ошибка: Файл модели/пайплайна '{model_file_path}' не найден.")
    print("Пожалуйста, проверьте правильность пути.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке модели/пайплайна: {e}")
    exit()

# --- 2. Загрузка объекта предобработки (ЕСЛИ загружена только модель и указан preprocessor_file_path) ---
if not is_pipeline and preprocessor_file_path:
    print("Загрузка объекта предобработки (загружена только модель)...")
    try:
        # Загружаем отдельно сохраненный препроцессор
        preprocessor = joblib.load(preprocessor_file_path)
        print(f"Объект предобработки успешно загружен из: {preprocessor_file_path}")

    except FileNotFoundError:
        print(f"Ошибка: Файл предобработки '{preprocessor_file_path}' не найден.")
        print("Если вы не сохраняли препроцессор отдельно или сохранили весь пайплайн, игнорируйте это сообщение.")
        print("Если вы использовали отдельный препроцессор, убедитесь, что путь правильный.")
        # Если препроцессор нужен, но не найден, прерываем выполнение
        if not is_pipeline:
             print("Невозможно выполнить предсказание без объекта предобработки, так как загружена только модель.")
             exit()
    except Exception as e:
        print(f"Ошибка при загрузке объекта предобработки: {e}")
        if not is_pipeline:
             print("Невозможно выполнить предсказание из-за ошибки загрузки препроцессора.")
             exit()


# --- Проверка: У нас должен быть либо пайплайн, либо модель И препроцессор ---
if not is_pipeline and preprocessor is None:
     print("\nКРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить ни полный пайплайн, ни модель вместе с отдельным препроцессором.")
     print("Убедитесь, что хотя бы один из сценариев загрузки сработал корректно.")
     exit()


# --- 3. Загрузка данных для предсказания (необработанных) ---
print("\nЗагрузка необработанных данных для предсказания...")
try:
    data_for_prediction = pd.read_csv(data_file_path)
    print(f"Данные успешно загружены из: {data_file_path}")
    print(f"Всего строк загружено: {len(data_for_prediction)}")
    # print("\nПервые 5 строк загруженных данных:")
    # print(data_for_prediction.head())
    # print("\nИнформация о столбцах и типах данных:")
    # data_for_prediction.info()
except FileNotFoundError:
    print(f"Ошибка: Файл данных '{data_file_path}' не найден.")
    print("Пожалуйста, проверьте правильность пути.")
    exit()
except Exception as e:
    print(f"Ошибка при загрузке данных: {e}")
    exit()

# --- 4. Подготовка данных для предсказания с помощью загруженного препроцессора/пайплайна ---

# Выбираем только те исходные признаки, которые нужны для предобработки/пайплайна
# Проверяем наличие всех необходимых исходных признаков в загруженных данных
missing_initial_features = [col for col in all_input_features if col not in data_for_prediction.columns]
if missing_initial_features:
    print(f"\nОшибка: В исходных данных для предсказания отсутствуют следующие признаки: {missing_initial_features}")
    print("Пожалуйста, убедитесь, что ваш CSV файл содержит все признаки, указанные в списках numerical_features и categorical_features.")
    exit()

# Выбираем исходные признаки в DataFrame
X_raw = data_for_prediction[all_input_features]

# Здесь не применяем transform, если это пайплайн. Пайплайн сделает это сам.
# Применяем transform только если у нас отдельный препроцессор.
X_processed = None # Переменная для хранения обработанных данных, если нужен отдельный препроцессор

if not is_pipeline:
    print("Выполнение предобработки данных отдельным препроцессором...")
    if preprocessor is not None:
        try:
             # Применяем преобразование с помощью загруженного препроцессора
            X_processed = preprocessor.transform(X_raw)
            print("Данные успешно предобработаны загруженным объектом препроцессора.")
        except Exception as e:
            print(f"Ошибка при выполнении предобработки: {e}")
            print("Убедитесь, что данные в '{data_file_path}' соответствуют ожиданиям препроцессора")
            print("(например, типы данных, отсутствие неожиданных категорий в категориальных признаках, если использовался OneHotEncoder).")
            exit()
    else:
        # Этого не должно произойти из-за проверки выше, но на всякий случай
        print("Ошибка: Объект предобработки не загружен, но загружена только модель.")
        exit()
else:
    print("Используется полный пайплайн. Предобработка будет выполнена автоматически при предсказании.")


# --- 5. Выполнение предсказаний ---
print("\nВыполнение предсказаний моделью/пайплайном...")
try:
    # Если это пайплайн, вызываем predict/predict_proba на нем, передавая НЕОБРАБОТАННЫЕ данные (X_raw)
    if is_pipeline:
        # predict_proba для логистической регрессии возвращает вероятности [класс 0, класс 1]
        if hasattr(model_or_pipeline, 'predict_proba'):
            probabilities = model_or_pipeline.predict_proba(X_raw)[:, 1]
            data_for_prediction['probability_default'] = probabilities
            print("Прогноз вероятности дефолта добавлен в столбец 'probability_default' (через пайплайн).")
        else:
             print("Пайплайн не имеет метода 'predict_proba'. Прогноз вероятности не добавлен.")

        if hasattr(model_or_pipeline, 'predict'):
            predicted_classes = model_or_pipeline.predict(X_raw)
            data_for_prediction['predicted_default_class'] = predicted_classes
            print("Прогноз класса дефолта добавлен в столбец 'predicted_default_class' (через пайплайн).")
        else:
             print("Пайплайн не имеет метода 'predict'. Прогноз класса не добавлен.")

    # Если это только модель (не пайплайн), вызываем predict/predict_proba на ней, передавая УЖЕ ОБРАБОТАННЫЕ данные (X_processed)
    elif model_or_pipeline is not None and X_processed is not None:
         if hasattr(model_or_pipeline, 'predict_proba'):
            probabilities = model_or_pipeline.predict_proba(X_processed)[:, 1]
            data_for_prediction['probability_default'] = probabilities
            print("Прогноз вероятности дефолта добавлен в столбец 'probability_default' (через модель и отдельный препроцессор).")
         else:
            print("Модель не имеет метода 'predict_proba'. Прогноз вероятности не добавлен.")

         if hasattr(model_or_pipeline, 'predict'):
            predicted_classes = model_or_pipeline.predict(X_processed)
            data_for_prediction['predicted_default_class'] = predicted_classes
            print("Прогноз класса дефолта добавлен в столбец 'predicted_default_class' (через модель и отдельный препроцессор).")
         else:
            print("Модель не имеет метода 'predict'. Прогноз класса не добавлен.")

    else:
        print("\nКРИТИЧЕСКАЯ ОШИБКА: Не удалось выполнить предсказания. Убедитесь, что загружены необходимые объекты и данные.")


except Exception as e:
    print(f"Ошибка при выполнении предсказаний: {e}")
    print("Убедитесь, что формат данных, поступающих в модель (после предобработки), соответствует тому, на чем модель обучалась.")
    # Продолжаем выполнение, чтобы сохранить хотя бы исходные данные

# --- 6. Сохранение результатов в новый CSV файл ---
print(f"\nСохранение результатов в файл: {output_file_path}...")
try:
    # Сохраняем DataFrame, который теперь содержит исходные столбцы + новые столбцы с предсказаниями
    data_for_prediction.to_csv(output_file_path, index=False)
    print(f"Результаты успешно сохранены в файл: {output_file_path}")
    print(f"Теперь файл '{output_file_path}' содержит ваши исходные данные плюс столбец(ы) с предсказаниями.")
except Exception as e:
    print(f"Ошибка при сохранении результатов: {e}")