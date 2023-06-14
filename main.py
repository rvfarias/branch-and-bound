from mip import *

def read_txt():
    sequencia = []
    arquivo = open('entrada.txt', 'r')
    for linha in arquivo:
        valores = linha.split()
        for v in valores:
            sequencia.append(v)
    
    return sequencia

def retorna_solucao(model, status):
    #print("Status = ", status)
    #print("Solution Value = ", model.objective_value)

    # for x in model.vars:
    #     print(f"{x.name} = {x.x}")
    return model.vars

def verifica_fracao(valor):
    return valor % 1 != 0

def escolhe_variavel(vars):
    valor_prox = None
    prox = float('inf')

    for v in vars:
        if verifica_fracao(v.x) == False:
            continue
        atual = abs(v.x - 0.5)
        if atual < prox:
            valor_prox = v.x
            prox = atual
            
        elif atual == prox:
            valor_prox = min(v.x, valor_prox)

    counter = 0
    if valor_prox == None:
        return -1
    
    for v in vars:
        if v.x == valor_prox:
            return counter

        counter +=1

def gateway():
    dados = read_txt()

    N_VARIAVEIS = dados[0]
    N_RESTRICOES = dados[1]
    del dados[0]
    del dados[0]

    coef_objetivo = []
    coef_restricoes = []
    dict_restricoes = {}

    for i in range(int(N_VARIAVEIS)):
        coef_objetivo.append(dados[0])
        del dados[0]

    print(f"N_VARIAVEIS = {N_VARIAVEIS} com N_RESTRICOES = {N_RESTRICOES}")

    for i in range(1,int(N_RESTRICOES)+1):
        for k in range(int(N_VARIAVEIS)+1):
            coef_restricoes.append(dados[0])
            del dados[0]
        
        dict_restricoes[f"R{i}"] = coef_restricoes
        coef_restricoes = []

    model = Model(sense=MAXIMIZE, solver_name=CBC)

    x = [model.add_var(var_type=CONTINUOUS, name=f"x_{d}", lb=0.0) for d in range(1,int(N_VARIAVEIS)+1)]
    model.objective = xsum(int(coef_objetivo[d])*x[(d)] for d in range(int(N_VARIAVEIS)))

    for i in range(1,int(N_RESTRICOES)):
        lhs = 0
        for k in range(int(N_VARIAVEIS)):
            lhs += int(dict_restricoes[f"R{i}"][k])*x[k]
        
        model += lhs <= int(dict_restricoes[f"R{i}"][int(N_VARIAVEIS)])

    for i in range(int(N_VARIAVEIS)):
        model += x[i] >= 0
        model += x[i] <= 1

    model.write("model.lp") # salva modelo em arquivo
    with open("model.lp", "r") as f: # lê e exibe conteúdo do arquivo
        print(f.read())

    status = model.optimize()
    valores = retorna_solucao(model, status)

    valor = escolhe_variavel(valores)

    if valor == -1:
        return primal
    
    model_1 = model.copy()
    contador = 0
    primal1, contador_nos1 = branch_and_bound(model, x, primal=0, restricao=0, valor=valor, contador_nos=contador)
    primal2, contador_nos2 = branch_and_bound(model_1, x, primal=primal1, restricao=1, valor=valor, contador_nos=contador_nos1)
    
    if primal1 >= primal2:
        primal = primal1
        return primal, model.vars, contador_nos2
    else:   
        primal = primal2
        return primal, model_1.vars, contador_nos2

def branch_and_bound(model, x, primal, restricao, valor, contador_nos):
    #print(f"x[{valor}] == {restricao} - primal = {primal}")
    global model_2
    model_2 = model.copy()
    
    model += x[valor] == restricao

    # model.write("model.lp") # salva modelo em arquivo
    # with open("model.lp", "r") as f: # lê e exibe conteúdo do arquivo
    #     print(f.read())

    contador_nos +=1

    status = model.optimize()
    valores = retorna_solucao(model, status)
    
    if model.objective_value == None:
        return primal, contador_nos
    elif model.objective_value <= primal:
        #print(f"menor que primal")
        return primal, contador_nos
    
    flag = True
    for i in valores:
        if verifica_fracao(i.x):
            flag = False
            break
    
    if flag:
        primal = model.objective_value
        return primal, contador_nos
    else:
        valor = escolhe_variavel(valores)
        if valor == -1:
            #print(f"valor = -1")
            return primal, contador_nos

        primal1, contador_nos = branch_and_bound(model, x, primal, 0, valor, contador_nos)
        print(f"Retornou do recurso para executar x[{valor+1}]")
        primal2, contador_nos  = branch_and_bound(model_2, x, primal1, 1, valor, contador_nos)

        tam = len(model_2.constrs) - 1
        restr = model_2.constrs[tam]
        model_2.remove(restr)

        if primal1 >= primal2:
            primal = primal1
        else:
            primal = primal2
    
    return primal , contador_nos

def main():
    primal, vars, contador_nos = gateway()
    print(f"\n\nMelhor primal encontrado = {primal}")
    print(f"Solução ótima!")
    print(f"Com as seguintes variáveis: \n")

    for i in vars:
        print(f"\t{i.name} = {i.x}")

    print(f"\nQuantidade de nós percorridos = {contador_nos+2}")

main()

