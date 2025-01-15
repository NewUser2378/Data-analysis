import re
import csv
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pymorphy2

file_path = 'uniqueLinks.txt'

# Открываем файл и читаем строки
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Удаляем символы новой строки и исключаем повторы с помощью множества
unique_urls = list(set(line.strip() for line in lines))

# Настройки браузера
options = uc.ChromeOptions()
options.add_argument('--headless')  # Запуск без графического интерфейса
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

# Открываем файл для записи в формате TSV
with open('good_ans.tsv', 'w', newline='', encoding='utf-8') as tsvfile, open('categories.txt', 'w', encoding='utf-8') as file1:
    # Создаем объект для записи данных в TSV
    writer = csv.DictWriter(tsvfile, fieldnames=[
        'URL', 'Характеристики', 'Детали квартиры'
    ], delimiter='\t')

    # Записываем заголовок таблицы
    writer.writeheader()

    # Используем множество для хранения уникальных признаков
    unique_features = {}

    # Используем контекстный менеджер для автоматического закрытия драйвера
    with uc.Chrome(driver_executable_path=ChromeDriverManager().install(), options=options) as driver:
        for url in unique_urls:
            try:
                driver.get(url)

                # Ожидаем появления элемента с нужным классом, максимальное время ожидания 3 секунды
                WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'OfferCardHighlight__value--HMVgP'))
                )

                # Получаем HTML-код страницы после полной загрузки
                html = driver.page_source

                # Парсим страницу с помощью BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')

                # Словарь для хранения данных объявления
                data = {
                    'URL': url + ';',
                    'Характеристики': '',
                }

                # Находим все элементы с классом OfferCardHighlight__label--2uMCy (категории)
                labels = soup.find_all('div', class_='OfferCardHighlight__label--2uMCy')
                # Находим все элементы с классом OfferCardHighlight__value--HMVgP (значения)
                elements = soup.find_all('div', class_='OfferCardHighlight__value--HMVgP')

                # Проверяем, нашлись ли такие элементы
                if labels and elements and len(labels) == len(elements):
                    details = []
                    for label, element in zip(labels, elements):
                        category = label.text.strip()  # Название категории
                        value = element.text.strip()  # Значение категории

                        # Разделяем значение по пробелам, чтобы отделить число от единицы измерения
                        value_parts = value.split()

                        if len(value_parts) > 1:
                            main_value = value_parts[0]  # Основное значение (например, "50")
                            unit = value_parts[1]  # Единица измерения (например, "м²")
                            # Формируем строку вида "Категория (ед. изм.): Значение"
                            if unit != 'этаж':
                                details.append(f"{category} ({unit}): {main_value}")
                                unique_features[f"{category} ({unit})"] = 'числовой'  # Числовой тип для значений с единицами измерения
                            else:
                                details.append(f"этаж: {main_value}/{re.sub(r'\D+', '', category)}")
                                unique_features["этаж"] = 'строковый'
                        else:
                            # Если единица измерения не найдена, предполагаем строковый тип
                            details.append(f"{category}: {value}")
                            unique_features[category] = 'строковый'

                # Поиск названия станции метро
                metro_station = soup.find('span', class_='MetroStation__title')
                if metro_station:
                    details.append(f"Станция метро: {metro_station.text.strip()}")
                    unique_features["Станция метро"] = 'строковый'

                # Поиск времени до метро
                time_to_metro = soup.find('span', string=lambda text: text and "мин." in text)
                if time_to_metro:
                    details.append(f"Расстояние до метро (мин): {time_to_metro.text.strip().split()[0]}")
                    unique_features["Расстояние до метро (мин)"] = 'числовой'

                # Поиск цены
                price = soup.find('span', class_='OfferCardSummaryInfo__price--2FD3C')
                if price:
                    price_text = price.text.strip()
                    price_number = re.findall(r'\d+', price_text)
                    price_cleaned = ''.join(price_number)
                    details.append(f"Стоимость (рубли): {price_cleaned}")
                    unique_features["Стоимость (рубли)"] = 'числовой'

                # Поиск типа квартиры
                apartment_type = soup.find('h1', class_='OfferCardSummaryInfo__description--3-iC7')
                if apartment_type:
                    apartment_text = apartment_type.text.strip()
                    apartment_cleaned = (apartment_text.split(','))[-1]
                    details.append(f"Тип квартиры: {apartment_cleaned}")
                    unique_features["Тип квартиры"] = 'строковый'

                # Поиск всех текстовых признаков из секции с классом OfferCardFeature__text--_Hmzv
                features = soup.find_all('div', class_='OfferCardFeature__text--_Hmzv')
                if features:
                    for feature in features:
                        feature_text = feature.text.strip()
                        index = feature_text.find(' нет')
                        index1 = feature_text.find('этаж')
                        index2 = feature_text.find('Дом')
                        index3 = feature_text.find('потолки')
                        index4 = feature_text.find('м²')
                        index5 = feature_text.find('Серия')
                        # Обрабатываем признаки и выводим в формате 1 или 0
                        if index1 == -1 and index2 == -1 and index3 == -1 and index4 == -1 and index5 == -1:
                            index_otdleka = feature_text.find('Отделка')
                            index_zdanie = feature_text.find('здание')
                            index_zastroysh = feature_text.find('Застройщик')
                            index_podezd = feature_text.find('подъезд')
                            index_kvartir = feature_text.find('кварт')
                            index_reconstr = feature_text.find('Реконструкция')
                            index_WC = feature_text.find('Санузел')
                            if index_otdleka != -1:
                                details.append(f"отделка: {feature_text}")
                                unique_features["отделка"] = 'строковый'
                            elif index_zastroysh != -1:
                                details.append(f"застройщик: {feature_text}")
                                unique_features["застройщик"] = 'строковый'
                            elif index_zdanie != -1:
                                details.append(f"тип здания: {feature_text}")
                                unique_features["тип здания"] = 'строковый'
                            elif index_WC != -1:
                                details.append(f"Санузел: {feature_text.split()[1]}")
                                unique_features["Санузел"] = 'строковый'
                            elif index_podezd != -1:  # числовые
                                details.append(f"число подъездов: {re.sub(r'\D+', '', feature_text)}")  # оставим только числа
                                unique_features["число подъездов"] = 'числовой'
                            elif index_kvartir != -1:  # числовые
                                details.append(f"число квартир: {re.sub(r'\D+', '', feature_text)}")
                                unique_features["число квартир"] = 'числовой'
                            elif index_reconstr != -1:
                                details.append(f"реконструкция: 1")
                                unique_features["реконструкция"] = 'бинарный'
                            elif index == -1:  # остальные признаки бинарные
                                details.append(f"{feature_text}: 1")  # Признак = 1
                                unique_features[feature_text] = 'бинарный'
                            else:
                                # не будем ничего делать так как признаки потом заменим на 0
                                x =1

                # Присоединяем все детали в строку
                data['Характеристики'] = "; ".join(details)

                # Записываем данные в файл TSV
                writer.writerow(data)

            except Exception as e:
                print(f"Произошла ошибка при работе с URL: {url}. Ошибка: {str(e)}")

    # Записываем уникальные категории и их типы в файл
    for key, value in unique_features.items():
        file1.write(f"{key}: {value}\n")
