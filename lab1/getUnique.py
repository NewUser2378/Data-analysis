# Для проверки, что все ОК и взяли нужное
input_file_path = 'links.txt'
output_file_path = 'uniqueLinks.txt'


with open(input_file_path, 'r', encoding='utf-8') as input_file:
    lines = input_file.readlines()

# Удаляем символы новой строки и исключаем повторы с помощью множества
unique_lines = set(line.strip() for line in lines)

# Открываем выходной файл и записываем уникальные строки
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in unique_lines:
        output_file.write(line + '\n')

print(f"Уникальные строки записаны в файл: {output_file_path}")
