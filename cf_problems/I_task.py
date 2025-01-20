#О, марковские цепи были у на на дискретке, есть идея что тк в марковских цепях у нас каждый следующий символ определяется с какой-то вероятностью
# а значит у нас какие-то пары символов будут встречаться чаше тк переходы в них как бы были с большей вероятностью
#в общем хочется найти такую строку, где число таких пар уникальных будем максимально отличаться от среднего и получится что там как бы нет закономерности

n = int(input())
strs = []
for i in range(n):
    strs.append(input().strip())

pairs_counts = []
for strr in strs:
    uni_pairs = set()
    for i in range(len(strr)-1):
        uni_pairs.add(strr[i:i+2])
    pairs_counts.append(len(uni_pairs))
mean = sum(pairs_counts)/n

ans = 0
cur =0
for i in range(n):
    if abs(pairs_counts[i]-mean) > cur:
        cur = (pairs_counts[i]-mean)
        ans = i

print(ans+1)
