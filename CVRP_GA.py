import random
from random import randrange
from time import time 
import numpy as np
import matplotlib.pyplot as plt



# La clase que representa el problema de cvrp con las funciones del algoritmo genetico 
# los atributos que se manejan son los siguientes:
# - genes: es una lista con los posibles nodos o clientes en el cromosoma 
# - individuals_length: longitud de la lista de genes o cromosoma 
# - decode: el metodo recive el cromosoma y regresa el decodificado con el respectivo nombre de los genes
# - fitness: metodo que devuelve la evaluacion del cromosoma 
# - mutation: realiza la mutacion de las posibles rutas 
# - crossover: Metodo para realizar el cruzamiento de los padres


class Problem_Genetic(object):
    
    def __init__(self,genes,individuals_length,decode,fitness):
        self.genes= genes
        self.individuals_length= individuals_length
        self.decode= decode
        self.fitness= fitness

    def mutation(self, chromosome, prob):
            
            def inversion_mutation(chromosome_aux):
                chromosome = chromosome_aux
                
                index1 = randrange(0,len(chromosome))
                index2 = randrange(index1,len(chromosome))
                
                chromosome_mid = chromosome[index1:index2]
                chromosome_mid.reverse()
                
                chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                
                return chromosome_result
        
            aux = []
            for _ in range(len(chromosome)):
                if random.random() < prob :
                    aux = inversion_mutation(chromosome)
            return aux

    def crossover(self,parent1, parent2):

        def process_gen_repeated(copy_child1,copy_child2):
            count1=0
            for gen1 in copy_child1[:pos]:
                repeat = 0
                repeat = copy_child1.count(gen1)
                if repeat > 1:#Si es necesario arreglar la generación en donde se repiten los nodos 
                    count2=0
                    for gen2 in parent1[pos:]:#Elegir la próxima generación disponible
                        if gen2 not in copy_child1:
                            child1[count1] = parent1[pos:][count2]
                        count2+=1
                count1+=1

            count1=0
            for gen1 in copy_child2[:pos]:
                repeat = 0
                repeat = copy_child2.count(gen1)
                if repeat > 1:#Si es necesario arreglar la generación en donde se repiten los nodos 
                    count2=0
                    for gen2 in parent2[pos:]:#CElegir la próxima generación disponible
                        if gen2 not in copy_child2:
                            child2[count1] = parent2[pos:][count2]
                        count2+=1
                count1+=1

            return [child1,child2]

        pos=random.randrange(1,self.individuals_length-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)
    
   
def decodeVRP(chromosome):    
    list=[]
    for (k,v) in chromosome:
        if k in trucks[:(num_trucks-1)]:
            list.append(frontier)
            continue
        list.append(cities.get(k))
        #print(list)
    return list


def penalty_capacity(chromosome):
        actual = chromosome
        value_penalty = 0
        capacity_list = []
        index_cap = 0
        overloads = 0
        
        for i in range(0,len(trucks)):
            init = 0
            capacity_list.append(init)
       
        for (k,v) in actual:
            if k not in trucks:
                capacity_list[int(index_cap)]+=v
            else:
                index_cap+= 1
                
            if  capacity_list[index_cap] > capacity_trucks:
                overloads+=1
                value_penalty+=  100 * overloads
        return value_penalty

def fitnessVRP(chromosome):
    
    def distanceTrip(index,city):
        w = distances.get(index)
        return  w[city]
        
    actualChromosome = chromosome
    fitness_value = 0
   
    penalty_cap = penalty_capacity(actualChromosome)
    for (key,value) in actualChromosome:
        if key not in trucks:
            nextCity_tuple = actualChromosome[key]
            if list(nextCity_tuple)[0] not in trucks:
                nextCity= list(nextCity_tuple)[0]
                fitness_value+= distanceTrip(key,nextCity) + (50 * penalty_cap)
                
    return fitness_value



# La función del algoritmo genetico (genetic_algoritm_t) recibe como entrada:
# * problem_genetic: una instancia de la clase Problem_Genetic, con
# el problema de optimización que queremos solucionar.
# * k: número de participantes en los torneos de selección.
# * opt es la el objetivo del problema a resolver para este caso se requiere minimizar el costo
# * ngen: número de generaciones (condición de parada)
# * size: número de individuos para cada generación
# * ratio_cross: porción de la población que será obtenida por
# medios de cruces.
# * prob_mutate: probabilidad de que se produzca una mutación genética. 


def genetic_algorithm_t(Problem_Genetic,k,opt,ngen,size,ratio_cross,prob_mutate):
    
    def initial_population(Problem_Genetic,size):   
        def generate_chromosome():
            chromosome=[]
            for i in Problem_Genetic.genes:
                chromosome.append(i)
            random.shuffle(chromosome)
            return chromosome
        
        return [generate_chromosome() for _ in range(size)]
            
    def new_generation_t(Problem_Genetic,k,opt,population,n_parents,n_directs,prob_mutate):
        
        def tournament_selection(Problem_Genetic,population,n,k,opt):
            winners=[]
            for _ in range(n):
                elements = random.sample(population,k)
                winners.append(opt(elements,key=Problem_Genetic.fitness))
            return winners
        
        def cross_parents(Problem_Genetic,parents):
            childs=[]
            for i in range(0,len(parents),2):
                childs.extend(Problem_Genetic.crossover(parents[i],parents[i+1]))
            return childs
    
        def mutate(Problem_Genetic,population,prob):
            for i in population:
                Problem_Genetic.mutation(i,prob)
            return population
                        
        directs = tournament_selection(Problem_Genetic, population, n_directs, k, opt)
        crosses = cross_parents(Problem_Genetic,
                                tournament_selection(Problem_Genetic, population, n_parents, k, opt))
        mutations = mutate(Problem_Genetic, crosses, prob_mutate)
        new_generation = directs + mutations
        
        return new_generation
    
    population = initial_population(Problem_Genetic, size)
    n_parents = round(size*ratio_cross)
    n_parents = (n_parents if n_parents%2==0 else n_parents-1)
    n_directs = size - n_parents
    
    for _ in range(ngen):
        population = new_generation_t(Problem_Genetic, k, opt, population, n_parents, n_directs, prob_mutate)
    
    bestChromosome = opt(population, key = Problem_Genetic.fitness)
    print("Cromosoma: ", bestChromosome)
    genotype = Problem_Genetic.decode(bestChromosome)
    print ("Solucion: " , (genotype,Problem_Genetic.fitness(bestChromosome)))
    return (genotype,Problem_Genetic.fitness(bestChromosome))


def CVRP(k):
    VRP_PROBLEM = Problem_Genetic([(0,0),(1,19),(2,16),(3,11),(4,15),(5,8),(6,8),(7,7),(8,14),(9,6),(10,11),
                                   (trucks[0],capacity_trucks)],
                                  len(cities), lambda x : decodeVRP(x), lambda y: fitnessVRP(y))
    
    
    def first_part_GA(k):
        cont  = 0
        
        print("Capacity of trucks = ",capacity_trucks)
        print("")
        tiempo_inicial_t2 = time()
        while cont <= k: 
            genetic_algorithm_t(VRP_PROBLEM, 4, min, 200, 100, 0.8, 0.05)
            cont+=1
        tiempo_final_t2 = time()
        print("\n") 
        print("Total time: ",(tiempo_final_t2 - tiempo_inicial_t2)," secs.\n")
    
    
    
    first_part_GA(k)
   


rnd = np.random
rnd.seed(0)


#CVRP Datos
n = 10  # numbre of clients
V = [0,1,2,3,4,5,6,7,8,9,10]
xc = [30,37,52,52,52,62,42,27,43,58,37]
yc = [40,52,64,33,41,42,57,68,67,48,69]

dc = [0,19,16,11,15,8,8,7,14,6,11]

print("Vertice",V)
print("X",xc)
print("Y",yc)
print("dc",dc)
print("Numero de vehiculos",2)

plt.plot(xc[0], yc[0], c='r', marker='s')
plt.scatter(xc[1:], yc[1:], c='b')

#Calculamos la Matriz de costos (distancia euclidiana de los nodos)

A = [(i, j) for i in V for j in V ]

C = [round(np.hypot(xc[i]-xc[j], yc[i]-yc[j]),2) for i, j in A]
C = np.array(C).reshape(11,11)
#Rutas (Solucion Basica inicial)
print(C)

#Ciudades (Represemtacion cromosomica )

cities = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'10'}


w0 = C[0]
w1 = C[1]
w2 = C[2]
w3 = C[3]
w4 = C[4]
w5 = C[5]
w6 = C[6]
w7 = C[7]
w8 = C[8]
w9 = C[9]
w10 = C[10]
distances = {0:w0,1:w1,2:w2,3:w3,4:w4,5:w5,6:w6,7:w7,8:w8,9:w9,10:w10}

capacity_trucks = 35

trucks = ['vehicle','vehicle']
num_trucks = len(trucks)
frontier = "-"

if __name__ == "__main__":
    # Constant that is an instance object 
    genetic_problem_instances = 10
    print("EXECUTING ", genetic_problem_instances, " INSTANCES ")
    CVRP(genetic_problem_instances)
    
