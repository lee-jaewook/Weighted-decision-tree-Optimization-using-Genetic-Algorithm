# Importing the libraries
import numpy as np
import geneticDicisionTree
import time
from tqdm import notebook

def generation(X_train,X_vali,y_train,y_vali):
    numberOfParents = 1000
    numberOfParentsMating = 200
    numberOfParameters = 7
    numberOfGenerations = 100
    # define the population size
    populationSize = (numberOfParents, numberOfParameters)
    # initialize the population with randomly generated parameters
    population = geneticDicisionTree.initilialize_poplulation(numberOfParents)
    # define an array to store the fitness  hitory
    fitnessHistory = np.empty([numberOfGenerations + 1, numberOfParents])
    # define an array to store the value of each parameter for each parent and generation
    populationHistory = np.empty([(numberOfGenerations + 1) * numberOfParents, numberOfParameters])
    # insert the value of initial parameters in history

    populationHistory[0:numberOfParents, :] = population

    for generation in notebook.tqdm(range(numberOfGenerations)):
        # print(" (%s / %s) 세대 개체 생성" % (generation+1,numberOfGenerations))
        fitnessValue = geneticDicisionTree.train_population(population=population,
                                                            X_train=X_train,
                                                            y_train=y_train,
                                                            X_vali=X_vali,
                                                            y_vali=y_vali)
        fitnessHistory[generation, :] = fitnessValue
        # print('Max G-measure in the this iteration = {}'.format(np.max(fitnessHistory[generation, :])))
        parents = geneticDicisionTree.new_parents_selection(population=population, fitness=fitnessValue, numParents=numberOfParentsMating)

        children = geneticDicisionTree.crossover_uniform(parents=parents, childrenSize=(populationSize[0] - parents.shape[0], numberOfParameters))

        children_mutated = geneticDicisionTree.mutation(children, numberOfParameters)

        population[0:parents.shape[0], :] = parents  # fittest parents
        population[parents.shape[0]:, :] = children_mutated  # children

        populationHistory[(generation + 1) * numberOfParents: (generation + 1) * numberOfParents + numberOfParents, :] = population
        time.sleep(0.01)
    # Best solution from the final iteration
    fitness = geneticDicisionTree.train_population(population=population,
                                                            X_train=X_train,
                                                            y_train=y_train,
                                                            X_vali=X_vali,
                                                            y_vali=y_vali)
    fitnessHistory[generation + 1, :] = fitness
    # index of the best solution
    bestFitnessIndex = np.where(fitness == np.max(fitness))[0][0]
    best_gene = np.empty([numberOfParameters,1])
    for i in range(numberOfParameters):
        best_gene[i]=population[bestFitnessIndex][i]
    print("-------------------------------------------------------")
    print("Best parameters are:")
    if population[bestFitnessIndex][0]==0:
        c_r='gini'
    else:
        c_r='entropy'
    print('criterion', c_r)
    print('max_depth', population[bestFitnessIndex][1])
    print('min_samples_split', int(population[bestFitnessIndex][2]))
    print('max_leaf_nodes', population[bestFitnessIndex][3])
    print('random_state', population[bestFitnessIndex][4])
    print('class_weight', population[bestFitnessIndex][5], population[bestFitnessIndex][6])
    print("-------------------------------------------------------")
    return fitness[bestFitnessIndex],best_gene