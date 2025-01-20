# О, индекс джинни уже считали в 4 лабе, идея простая просто разбиваем на две части и считаем индекс джинни для двух частей пропорцианально их размеру
# и чем он меньше тем лучше разбиение, так как он оценивает вероятность,
# что случайный объект неправильно классифицирован если класс предсказывается случайно на основе нашего разбиения
# тк получим сумму 1-Sum(pi)^2 и так по всем классам для левой и правой частей
# считать буду как в 4 лабе только сейчас нам даны классы и сделаем отдельные массивы для разбиения на левую и правую части
# сначала кладем все направо и перекладываем поочередно налево


n, k = map(int, input().split())
classes = list(map(int, input().split()))

left_counts = [0] * (k + 1)
right_counts = [0] * (k + 1)
left_size = 0
right_size = n
ans = []

for cl in classes:
    right_counts[cl] += 1

for i in range(n - 1):
    cur_class = classes[i]
    right_size -= 1
    left_size += 1
    left_counts[cur_class] += 1
    right_counts[cur_class] -= 1

    sum1 = 0
    sum2 = 0
    for cl_count in left_counts:
        sum1 += (cl_count / left_size) ** 2
    for cl_count in right_counts:
        if right_size > 0:
            sum2 += (cl_count / right_size) ** 2

    gini1 = 1 - sum1
    gini2 = 1 - sum2
    gini_ans = (left_size / n) * gini1 + (right_size / n) * gini2
    ans.append(gini_ans)

print("\n".join(f"{gini:.9f}" for gini in ans))
