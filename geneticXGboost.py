import numpy as np
import pandas as pd
import random
from sklearn.metrics import f1_score
import xgboost as xgb

def initilialize_poplulation(numberOfParents):
    learningRate = np.empty([numberOfParents, 1])
    nEstimators = np.empty([numberOfParents, 1], dtype=np.uint8)
    maxDepth = np.empty([numberOfParents, 1], dtype=np.uint8)
    minChildWeight = np.empty([numberOfParents, 1])
    gammaValue = np.empty([numberOfParents, 1])
    subSample = np.empty([numberOfParents, 1])
    colSampleByTree = np.empty([numberOfParents, 1])
    for i in range(numberOfParents):
        print(i)
        learningRate[i] = round(random.uniform(0.01, 1), 2)
        nEstimators[i] = random.randrange(10, 1500, step=25)
        maxDepth[i] = int(random.randrange(1, 10, step=1))
        minChildWeight[i] = round(random.uniform(0.01, 10.0), 2)
        gammaValue[i] = round(random.uniform(0.01, 10.0), 2)
        subSample[i] = round(random.uniform(0.01, 1.0), 2)
        colSampleByTree[i] = round(random.uniform(0.01, 1.0), 2)

    population = np.concatenate((learningRate, nEstimators, maxDepth, minChildWeight, gammaValue, subSample, colSampleByTree), axis=1)
    return population

def fitness_f1score(y_true, y_pred):
    fitness = round((f1_score(y_true, y_pred, average='weighted')), 4)
    return fitness

def train_population(population, dMatrixTrain, dMatrixtest, y_test):
    fScore = []
    for i in range(population.shape[0]):
        param = { 'objective':'binary:logistic',
                  'learning_rate': population[i][0],
                  'n_estimators': population[i][1],
                  'max_depth': int(population[i][2]),
                  'min_child_weight': population[i][3],
                  'gamma': population[i][4],
                  'subsample': population[i][5],
                  'colsample_bytree': population[i][6],
                  'seed': 24}
        num_round = 100
        xgbT = xgb.train(param, dMatrixTrain, num_round)
        preds = xgbT.predict(dMatrixtest)
        preds = preds>0.5
        fScore.append(fitness_f1score(y_test, preds))
    return fScore

def new_parents_selection(population, fitness, numParents):
    selectedParents = np.empty((numParents, population.shape[1]))  # create an array to store fittest parents

    # find the top best performing parents
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[
            bestFitnessId] = -1  # set this value to negative, in case of F1-score, so this parent is not selected again
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
    minMaxValue[0:] = [0.01, 1.0]  # min/max learning rate
    minMaxValue[1, :] = [10, 2000]  # min/max n_estimator
    minMaxValue[2, :] = [1, 15]  # min/max depth
    minMaxValue[3, :] = [0, 10.0]  # min/max child_weight
    minMaxValue[4, :] = [0.01, 10.0]  # min/max gamma
    minMaxValue[5, :] = [0.01, 1.0]  # min/maxsubsample
    minMaxValue[6, :] = [0.01, 1.0]  # min/maxcolsample_bytree

    # Mutation changes a single gene in each offspring randomly.
    mutationValue = 0
    parameterSelect = np.random.randint(0, 7, 1)
    print(parameterSelect)
    if parameterSelect == 0:  # learning_rate
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 1:  # n_estimators
        mutationValue = np.random.randint(-200, 200, 1)
    if parameterSelect == 2:  # max_depth
        mutationValue = np.random.randint(-5, 5, 1)
    if parameterSelect == 3:  # min_child_weight
        mutationValue = round(np.random.uniform(5, 5), 2)
    if parameterSelect == 4:  # gamma
        mutationValue = round(np.random.uniform(-2, 2), 2)
    if parameterSelect == 5:  # subsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)
    if parameterSelect == 6:  # colsample
        mutationValue = round(np.random.uniform(-0.5, 0.5), 2)

# indtroduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if (crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 1]
        if (crossover[idx, parameterSelect] < minMaxValue[parameterSelect, 0]):
            crossover[idx, parameterSelect] = minMaxValue[parameterSelect, 0]
    return crossover