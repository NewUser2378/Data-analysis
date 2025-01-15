import pandas as pd

def process_realty_data(input_file, output_file):
    df = pd.read_csv(input_file)

    # Заменяем знаки вопроса на нули
    df = df.fillna(0)
    df = df.replace("?", 0)

    # будем считать процент для этажа, то есть насколько он будет хорошо расположен относительно всех вариантов, для этого выражение вида "9/25" парсим в 9/25
    df['этаж'] = df['этаж'].apply(lambda x: eval(x.replace(' ', '').replace('/', '/')) if '/' in str(x) else float(x))

    # (подсчитываем только ненулевые и длиной хотя бы 2 символа) идея, чтобы он показывал
    df['застройщик'] = df['застройщик'].apply(lambda x: x if len(str(x)) >= 2 else None)  # Заменяем значения длиной менее 2 на None
    df['застройщик'] = df.groupby('застройщик')['застройщик'].transform('count')  # Подсчитываем количество вхождений
    df['застройщик'] = df['застройщик'].fillna(0)  # Заполняем None обратно нулями

    # Обработка признака "Тип здания"
    df['тип здания'] = df['тип здания'].replace({
        'Монолитное здание': 5,
        'Кирпичное здание': 3,
        'Кирпично-монолитное здание': 4,
        'Панельное здание': 2
    }).fillna(0)  # Если не указано, ставим 0


    df['Тип квартиры'] = df['Тип квартиры'].apply(lambda x: 0.5 if 'студия' in str(x).lower() else float(x.split('-')[0]) if '-' in str(x) else 0)


    df['отделка'] = df['отделка'].replace({
        'Отделка — дизайнерский ремонт': 3,
        'Отделка — евроремонт': 2,
        'Отделка — косметический ремонт': 1,
        'Отделка — требуется ремонт': -1
    }).fillna(0)  # Если отделка не указана, ставим 0

    df['Санузел'] = df['Санузел'].replace({
        'раздельный': 1,
        'совмещённый': 2
    }).fillna(0)  # Если не указано, ставим 0

    # Обработка признака "Число квартир" (заменяем 0 на 1)
    df['число квартир'] = df['число квартир'].replace(0, 1).fillna(1)  # Если 0 или отсутствует значение, ставим 1

    # Выбираем станцию метро как целевой признак
    target_column = 'Станция метро'
    target = df[target_column]
    df.drop(columns=[target_column], inplace=True)

    # Сохраняем результат в CSV
    df[target_column] = target  # Добавляем целевой признак в конец
    df.to_csv(output_file, index=False)
    print(f"Файл успешно сохранен: {output_file}")

if __name__ == "__main__":
    input_csv_file = 'realty_data.csv'
    output_csv_file = 'normed_data.csv'
    process_realty_data(input_csv_file, output_csv_file)
