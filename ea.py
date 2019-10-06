import random as rand;
import math as math;
import numpy as np;
#hyperparameters of EA


CROSSOVER_PROB = 0.5 #0.7 is recommended though in the referred paper
FITNESS_EVALUATOR_GENERATION_COUNT = 5;
WEIGHT_VECTOR_DIMENSION = 30 #when I pulled, the weight vector was 30
TOURNAMENT_SIZE = 10 #The larger the tournament size, the great is the probability of loss of diversity ref: https://goo.gl/8LsvCy      pg no: 4
HEURISTIC_CUSTOM_SELECTION_COUNT = 15;


class Gene:

    def __init__(self, id, weightVector = np.array([]), fitness = math.inf): #Intialize the object and also give it a rating -

        self.id = id;

        #These if else block is just to make sure that when creating multiple instances of the object
        #they won't end up having the same random values

        if(weightVector.size==0):
            self.weightVector = np.random.rand(WEIGHT_VECTOR_DIMENSION,1)*2 - 1;
        else:
           self.weightVector = weightVector;

        self.fitness = fitness;



    def crossOver(self, p2):  # additive cross over

        """
        :param self: represents the object calling this function or simply the first parent
        p2: represents the second parent.
        :return: The function returns two child after crossing the binary weight vectors based on a crossoverpoint
        """
        # defined by the hapt's method ref: https://goo.gl/q4FFrx
        c1WeightVector, c2WeightVector = self.haptCrossOver(p2);

        #all method below are from paper https://goo.gl/dGECKW
        #c1WeightVector, c2WeightVector = self.pointCrossOver(p2,1); #single point cross over
        #c1WeightVector, c2WeightVector = self.pointCrossOver(p2,2); #2 point cross over
        #c1WeightVector, c2WeightVector = self.uniformCrossOver(p2);
        #c1WeightVector, c2WeightVector = self.flatCrossOver(p2);
        return Gene(0, c1WeightVector), Gene(0,c2WeightVector); #return two children or offsprings

    def haptCrossOver(self, p2):
        p1 = self;

        # defined by the hapt's method ref: https://goo.gl/q4FFrx
        alpha = rand.randint(0,WEIGHT_VECTOR_DIMENSION-1); #cross over point

        beta = rand.random(); #a random value between 0 and 1

        p1wAlpha = p1.weightVector[alpha] - beta * (p1.weightVector[alpha]-p2.weightVector[alpha]);
        p2wAlpha = p2.weightVector[alpha] - beta * (p1.weightVector[alpha] - p2.weightVector[alpha]);

        c1WeightVector = p1.weightVector;
        c2WeightVector = p2.weightVector;

        c1WeightVector[alpha] = p1wAlpha;
        c2WeightVector[alpha] = p2wAlpha;

        return c1WeightVector,c2WeightVector;

    def pointCrossOver(self, p2, num_points=1):
        p1 = self;

        c1WeightVector = p1.weightVector;
        c2WeightVector = p2.weightVector;

        cross_point1 = rand.randint(0, WEIGHT_VECTOR_DIMENSION - 1);
        cross_point2 = WEIGHT_VECTOR_DIMENSION-1;

        if(num_points==2):
            cross_point2 = rand.randint(cross_point1, WEIGHT_VECTOR_DIMENSION - 1);

        for index in range(cross_point1,cross_point2+1):
            c1WeightVector[index]=p2.weightVector[index];
            c2WeightVector[index]=p1.weightVector[index];

        return c1WeightVector,c2WeightVector;

    def uniformCrossOver(self, p2):

        p1 = self;
        c1WeightVector = c2WeightVector = np.array([]);

        for i in range(WEIGHT_VECTOR_DIMENSION):
            prob = rand.uniform(-1,1);
            if(prob < CROSSOVER_PROB):
                c1WeightVector = np.append(c1WeightVector, p1.weightVector[i]);
                c2WeightVector = np.append(c2WeightVector, p2.weightVector[i]);
            else:
                c1WeightVector = np.append(c1WeightVector, p2.weightVector[i]);
                c2WeightVector = np.append(c2WeightVector, p1.weightVector[i]);

        c1WeightVector = np.reshape(c1WeightVector,(WEIGHT_VECTOR_DIMENSION,1));
        c2WeightVector = np.reshape(c2WeightVector,(WEIGHT_VECTOR_DIMENSION,1));

        return c1WeightVector,c2WeightVector;

    def flatCrossOver(self, p2):

        p1=self;
        randVector1 = np.random.rand(WEIGHT_VECTOR_DIMENSION,1);
        randVector2 = np.random.rand(WEIGHT_VECTOR_DIMENSION,1);

        c1WeightVector = c2WeightVector = np.array([]);

        for i in range(WEIGHT_VECTOR_DIMENSION):
            c1WeightVector = np.append(c1WeightVector, randVector1[i]*p1.weightVector[i]+(1-randVector1[i])*p2.weightVector[i]);
            c2WeightVector = np.append(c2WeightVector, randVector2[i] * p1.weightVector[i] + (1 - randVector2[i]) * p2.weightVector[i]);

        c1WeightVector = np.reshape(c1WeightVector, (WEIGHT_VECTOR_DIMENSION, 1));
        c2WeightVector = np.reshape(c2WeightVector, (WEIGHT_VECTOR_DIMENSION, 1));

        return c1WeightVector,c2WeightVector;


    def mutate(self):
        """
        when a parameter in a chromosome mutates, its value is replaced with a random number between 0 and 10
        This is a method specified in the paper that chinmay shared ref:pg no: 15 on the top.
        :return: mutated gene
        """
        index = rand.randint(0,WEIGHT_VECTOR_DIMENSION-1);
        self.weightVector[index] = rand.uniform(0,1)
        return self;  # returning mutated gene

class Generation:

    def __init__(self, population_size, selectionCount):
        self.population_size = population_size;
        self.selectionCount = selectionCount;
        self.genes = [Gene(i) for i in range(population_size)]; #population


    def selection(self):
        """
        Selection methods used are from the literature https://goo.gl/8LsvCy
        :return:
        """


        self.genes.sort(key=lambda node: node.fitness, reverse = True);
        heuristicGenes = self.genes[0:HEURISTIC_CUSTOM_SELECTION_COUNT];
        selectedGenes = self.tournamentSelection();

        #selectedGenes  = self.rouletteWheelSelection();
        #selectedGenes  = self.linearRankingSelection();
        self.genes = selectedGenes + heuristicGenes;


    def tournamentSelection(self):
        selectedGenes = [];
        for i in range(self.selectionCount):
            indeces = rand.sample(range(0, self.population_size - 1), TOURNAMENT_SIZE);
            competitors = [];
            for index in indeces:
                competitors.append(self.genes[index]);
            competitors.sort(key=lambda node: node.fitness, reverse=True);
            selectedGenes.append(competitors[0]);
        return selectedGenes;

    def rouletteWheelSelection(self): #proportionate selection

        """
        ref: https://goo.gl/YVY7nU
        :return:
        """
        sum = 0;
        wheel = [];


        for gene in self.genes:
            sum += gene.fitness;

        prob = 0;
        for gene in self.genes:
            prob += gene.fitness / sum;
            wheel.append(prob);  # placing in the wheel as sectors based on the probability

        return self.runWheelandSelectGenes(wheel);


    def linearRankingSelection(self): #rank based selection
        """
        ref:https://goo.gl/G3TpLC
        :return:
        """
        wheel = [];
        self.genes.sort(key=lambda node: node.fitness, reverse=True);  # sorting the genes

        rank = self.population_size;
        prob = 0;
        sum = (self.population_size * (self.population_size + 1)) / 2;  # gauss law

        for gene in self.genes:
            prob += rank / sum;
            wheel.append(prob);  # placing in the wheel as sectors based on the probability
            rank -= 1;

        return self.runWheelandSelectGenes(wheel);

    def runWheelandSelectGenes(self, wheel):
        selectedGenes = []
        for i in range(self.selectionCount):
            spin_point = rand.random(); #equivalent to spinning the wheel
            for index in range(len(wheel)):
                if (spin_point <= wheel[index]):
                    selectedGenes.append(self.genes[index]);
                    break;
        return selectedGenes;

    def reproduction(self):  # cross over is doing here
        x = 0;
        y = 1;
        #for i in range(self.population_size):
        for i in range(int((self.population_size - (self.selectionCount+HEURISTIC_CUSTOM_SELECTION_COUNT))/2)):
            #i = rand.randint(0, self.selectionCount - 1);
            #j = rand.randint(0, self.selectionCount - 1)
            childs = self.genes[x].crossOver(self.genes[y]);
            x += 2;
            y += 2;

            for child in childs:
                self.genes.append(child);

    def mutation(self, prob):
        """
        ref pg no:5 https://goo.gl/trNAz2
        :param prob:
        :return:
        """
        mutation_count = math.ceil(prob * self.population_size);

        indeces = rand.sample(range(0,self.population_size-1),mutation_count);
        for index in indeces:
            self.genes[index] = self.genes[index].mutate();

