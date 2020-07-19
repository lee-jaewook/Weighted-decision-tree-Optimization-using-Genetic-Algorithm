import numpy as np
import pandas as pd
import random
import g_measure
from sklearn.tree import DecisionTreeClassifier


def initilialize_poplulation(numberOfParents):
    criterion = np.empty([numberOfParents, 1], dtype=np.uint8)
    maxDepth = np.empty([numberOfParents, 1], dtype=np.uint8)
    minSamplesSplit = np.empty([numberOfParents, 1], dtype=np.uint8)
    maxLeafNodes = np.empty([numberOfParents, 1], dtype=np.uint8)
    randomState = np.empty([numberOfParents, 1])
    trueWeight = np.empty([numberOfParents, 1])
    falseWeight = np.empty([numberOfParents, 1])
    print("Initilialize Poplulation",end='')
    for i in range(numberOfParents):
        # print(i)
        criterion[i] = int(random.randrange(0, 2, step=1))
        maxDepth[i] = int(random.randrange(1, 20, step=1))
        minSamplesSplit[i] = int(random.randrange(2, 10, step=1))
        maxLeafNodes[i] = int(random.randrange(10, 1500, step=25))
        randomState[i] = int(random.randrange(10, 1000, step=20))
        trueWeight[i] = round(random.uniform(0.01, 1), 2)
        falseWeight[i] = round(random.uniform(0.01, 1), 2)
        # print("***Initilialize Poplulation***")
        # print("<최초 %s번째 유전자 생성> Criterion: %s | Max Depth: %s | min samples split: %s | max leaf nodes: %s | randomstate: %s | true-false weight: %s/%s"%(i+1,criterion[i],maxDepth[i],minSamplesSplit[i],maxLeafNodes[i],randomState[i],trueWeight[i],falseWeight[i]))
    population = np.concatenate(
        (criterion, maxDepth, minSamplesSplit, maxLeafNodes, randomState, trueWeight, falseWeight), axis=1)
    print("------> Complete")
    return population

def fitness_Gmeasure(y_true, y_pred):
    fitness = round((g_measure.score(y_true, y_pred)), 4)
    return fitness

def train_population(population, X_train, y_train, X_vali, y_vali):
    g_measure = []

    for i in range(population.shape[0]):
        param = { 'criterion': population[i][0],
                  'max_depth': population[i][1],
                  'min_samples_split':population[i][2],
                  'max_leaf_nodes':population[i][3],
                  'random_state':population[i][4],
                  'true_weight':population[i][5],
                  'false_weight':population[i][6]}

        if param['criterion']==0:
            cr = 'gini'
        else:
            cr = 'entropy'

        m_d = int(param['max_depth'])
        m_s_s = int(param['min_samples_split'])
        m_l_n = int(param['max_leaf_nodes'])
        r_s = int(param['random_state'])
        balance = {True:param['true_weight'],False:param['false_weight'] }
        clf = DecisionTreeClassifier(criterion=cr,
                                     max_depth=m_d,
                                     min_samples_split=m_s_s,
                                     max_leaf_nodes=m_l_n,
                                     random_state=r_s,
                                     class_weight=balance
                                     )

        clf.fit(X_train, y_train)
        preds = clf.predict(X_vali)
        g_measure.append(fitness_Gmeasure(y_vali, preds))
    return g_measure

def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1]))  # create an array to store fittest parents

    # find the top best performing parents
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1  # set this value to negative, in case of F1-score, so this parent is not selected again
    return selectedParents


def crossover_uniform(parents, childrenSize):
    crossoverPointIndex = np.arange(0, np.uint8(childrenSize[1]), 1, dtype=np.uint8)  # get all the index
    crossoverPointIndex1 = np.random.randint(0, np.uint8(childrenSize[1]),
                                             np.uint8(childrenSize[1] / 2))  # select half  of the indexes randomly
    crossoverPointIndex2 = np.array(
        list(set(crossoverPointIndex) - set(crossoverPointIndex1)))  # select leftover indexes

    children = np.empty(childrenSize)

    '''
    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
    will be picked from the indexes, which were randomly selected above. 
    '''
    for i in range(childrenSize[0]):
        # find parent 1 index
        parent1_index = i % parents.shape[0]
        # find parent 2 index
        parent2_index = (i + 1) % parents.shape[0]
        # insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex1] = parents[parent1_index, crossoverPointIndex1]
        # insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex2] = parents[parent2_index, crossoverPointIndex2]
    return children


def mutation(crossover, numberOfParameters):
# Define minimum and maximum values allowed for each parameter
    minMaxValue = np.zeros((numberOfParameters, 2))
    minMaxValue[0, :] = [0, 1]  # criterion
    minMaxValue[1, :] = [2, 20]  # min/max max depth
    minMaxValue[2, :] = [2, 10]  # min/max min sample split
    minMaxValue[3, :] = [10, 1500]  # min/max max leaf nodes
    minMaxValue[4, :] = [10, 1000]  # min/max random state
    minMaxValue[5, :] = [0.01, 1.0]  # min/max true weight
    minMaxValue[6, :] = [0.01, 1.0]  # min/max false weight

    # Mutation changes a single gene in each offspring randomly.
    mutationValue = 0
    parameterSelect = np.random.randint(0, 7, 1)
    # print(parameterSelect)
    if parameterSelect == 0:  # criterion
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 1:  # max depth
        mutationValue = np.random.randint(-10, 10, 1)
    if parameterSelect == 2:  # min sample split
        mutationValue = np.random.randint(-5, 5, 1)
    if parameterSelect == 3:  # max leaf nodes
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 4:  # random state
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 5:  # true weight
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 6:  # false weight
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)

# indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if (crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
        if (crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]
    return crossover