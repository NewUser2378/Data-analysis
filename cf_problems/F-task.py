# опять начну с идеи, у нас есть матрица ошибок, на диагонали у нас элементы которые правильно классифицировали
# а в остальных в клетке ij количество из класса i которые классифицировали как j
# для метрик будем считать число правильных, неправильных и пропущенныных обозначим как на лекции tp fp fn
#в общем идея что мы для каждого класса i считаем точность как precision_i = tp_i/(tp_i+fp_i), чтобы посмотреть как часто у нас верные прогнозы по сути получается что-то типа вероятности
#есть еще recall для каждого класса i чтобы посмотреть на то часто ли пропускаем объекты этого класса, считаем аналогично и получается тоже похоже на вероятность recall_i = tp_i/(tp_i+fn_i)
# теперь именно про меры, начну с не взвешенного F1 в для одного класса смотрим на precision_i и recall_i чтобы если одна из метрик низкая то F1 тоже поэтому считаем как 2*precision_i * recall_i/(precision_i + recall_i)
# в взвешенной F1 просто нормируем с учетом весов: w_i = fn_i + tp_i и тогда в итоге получим Sum w_i*F1_i/Sum w_i
# микро считаем аналогично только precision и recall считаем по всем tp np fp а потом по формуле как F1 ну то есть считаем не по классам а по всем значениям, то есть как бы находим полноту и точность для одной большой бинпрной задачи
# в макро просто точность и полноту считаем для каждого класса отдельно, а затем усредняем с учетом веса
# в коде посчитаем по формула и если где то 0 в знаменателе то считаем значение = 0

k = int(input())
conf_matrix = [0] * (k * k)
tp = [0] * k
fp = [0] * k
fn = [0] * k
n = 0

for row in range(k):
    vals = list(map(int, input().split()))
    for col in range(k):
        count = vals[col]
        conf_matrix[k * row + col] = count
        n += count

for row in range(k):
    for col in range(k):
        if row != col:
            fn[row] += conf_matrix[row * k + col]
            fp[row] += conf_matrix[col * k + row]
    tp[row] = conf_matrix[row * k + row]

micro_tp = 0
for i in range(k):
    micro_tp+= tp[i] * (fn[i] + tp[i])
micro_tp/= n

micro_fp = 0
for i in range(k):
    micro_fp+= fp[i] * (fn[i] + tp[i])
micro_fp/= n


micro_fn = 0
for i in range(k):
    micro_fn+= fn[i] * (fn[i] + tp[i])
micro_fn/= n


macro_prec = 0
for i in range(k):
    macro_prec += (tp[i] / (tp[i] + fp[i])) * (fn[i] + tp[i])
macro_prec /= n

macro_rec = 0
for i in range(k):
    macro_rec += (tp[i] / (tp[i] + fn[i])) * (fn[i] + tp[i])
macro_rec /= n

if (micro_tp + micro_fp) == 0:
    micro_prec = 0
else:
    micro_prec = micro_tp / (micro_tp + micro_fp)
if (micro_tp + micro_fp) == 0:
    micro_rec = 0
else:
    micro_rec = micro_tp / (micro_tp + micro_fn)

if (micro_prec + micro_rec) == 0:
    micro_f1 = 0
else:
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec)

if (macro_prec + macro_rec) == 0:
    macro_prec = 0
if (macro_prec + macro_rec) == 0:
    macro_rec = 0
if (macro_prec + macro_rec) == 0:
    macro_f1 = 0
else:
    macro_f1 = 2 * macro_prec * macro_rec / (macro_prec + macro_rec)


base_f1 = 0
for i in range(k):
    if (tp[i] + fp[i]) == 0:
        prec = 0
    else:
        prec = tp[i] / (tp[i] + fp[i])
    if (tp[i] + fn[i]) == 0:
        rec = 0
    else:
        rec = tp[i] / (tp[i] + fn[i])
    if prec + rec > 0:
        base_f1 += (2 * prec * rec / (prec + rec)) * (fn[i] + tp[i])
base_f1 /= n

print(micro_f1)
print(macro_f1)
print(base_f1)
