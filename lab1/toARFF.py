import csv
import pandas as pd


def read_categories(file_path):
    """Читаем файл categories.txt и возвращаем словарь признаков и их типов."""
    categories = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ": " in line:
                feature, feature_type = line.strip().split(": ", 1)
                categories[feature.strip()] = feature_type.strip()
    return categories


def parse_good_ans(file_path):
    """Читаем файл good_ans.tsv и возвращаем список словарей с признаками и значениями."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            url = row[0]  # URL объекта недвижимости
            features_string = row[1]  # Строка с признаками
            feature_dict = {"URL": url}

            # Разделяем строку признаков по разделителю ";"
            features = features_string.split(";")
            for feature in features:
                if ": " in feature:
                    key, value = feature.split(": ", 1)
                    feature_dict[key.strip()] = value.strip()

            data.append(feature_dict)
    return data


def convert_to_arff(data, categories, output_file):
    """Преобразуем данные в формат ARFF и записываем их в файл."""

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("@RELATION realty_data\n\n")

        # Записываем атрибуты и их типы
        for feature, feature_type in categories.items():
            feature_name = feature.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')

            if feature_type == "числовой":
                file.write(f"@ATTRIBUTE {feature_name} NUMERIC\n")
            elif feature_type == "строковый":
                file.write(f"@ATTRIBUTE {feature_name} STRING\n")
            elif feature_type == "бинарный":
                file.write(f"@ATTRIBUTE {feature_name} {{0, 1}}\n")

        # Записываем URL как строковый атрибут
        file.write("@ATTRIBUTE URL STRING\n\n")

        file.write("@DATA\n")

        # Записываем сами данные
        for entry in data:
            row = []
            for feature in categories:
                value = entry.get(feature, "?")  # Используем "?" если значения нет

                if categories[feature] == "числовой":
                    try:
                        row.append(float(value))  # Числовое значение
                    except ValueError:
                        row.append("?")
                elif categories[feature] == "строковый":
                    row.append(f"'{value}'")  # Строковые значения
                elif categories[feature] == "бинарный":
                    row.append('1' if value == '1' else '0')  # Бинарные значения: 1 или 0

            # Добавляем URL в конец строки
            row.append(f"'{entry['URL']}'")
            file.write(",".join(map(str, row)) + "\n")


def convert_to_csv(data, categories, output_file):
    """Преобразуем данные в формат CSV и записывает их в файл."""
    with open(output_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)

        # Записываем заголовки
        header = list(categories.keys()) + ["URL"]
        writer.writerow(header)

        # Записываем данные
        for entry in data:
            row = []
            for feature in categories:
                value = entry.get(feature, "?")
                row.append(value)
            row.append(entry['URL'])
            writer.writerow(row)


if __name__ == "__main__":
    good_ans_file = 'good_ans.tsv'
    categories_file = 'categories.txt'
    output_arff_file = 'realty_data.arff'
    output_csv_file = 'realty_data.csv'

    # получаем данные из файлов
    categories = read_categories(categories_file)
    data = parse_good_ans(good_ans_file)

    # Преобразуем данные в ARFF и записываем в файл
    convert_to_arff(data, categories, output_arff_file)

    # Преобразуем данные в CSV и записываем в файл
    convert_to_csv(data, categories, output_csv_file)

    print(f"Файлы ARFF и CSV были успешно созданы: {output_arff_file}, {output_csv_file}")
