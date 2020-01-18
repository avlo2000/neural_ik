n = int(input())
answs = []
for i in range(n):
    sen = input()
    if(sen[-2 :] == "po"):
        answs.append("FILIPINO")
    elif (sen[-4 :] == "desu" or sen[-4 :] == "masu"):
        answs.append("JAPANESE")
    elif (sen[-5:] == "mnida", sen[-5:] == "mnida"):
        answs.append("KOREAN")

for a in answs:
    print(a)