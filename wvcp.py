import random
import sys
import os
import time

class MWVCInstance:
    """
    Classe che rappresenta un'istanza del problema WVCP.
    """
    def __init__(self, filepath=None, raw_data=None):
        self.num_nodes = 0
        self.weights = []
        self.adj_matrix = []
        self.edges = []
        self.adj_list = [] # Lista di adiacenza per calcoli super-veloci
        
        if filepath:
            with open(filepath, 'r') as f:
                self._parse_data(f.read())
        elif raw_data:
            self._parse_data(raw_data)
        else:
            raise ValueError("fornire un filepath o raw_data.")

    def _parse_data(self, raw_data):
        lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
        self.num_nodes = int(lines[0])
        self.weights = [float(w) for w in lines[1].split()]
        
        for i in range(2, 2 + self.num_nodes):
            row = [int(val) for val in lines[i].split()]
            self.adj_matrix.append(row)
            
        self.adj_list = [[] for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adj_matrix[i][j] == 1:
                    self.edges.append((i, j))
                    self.adj_list[i].append(j)
                    self.adj_list[j].append(i)
                    
        print(f"Istanza caricata con successo: {self.num_nodes} nodi, {len(self.edges)} archi.")


class Immune_Inspired:
    """
    Algoritmo basato sul Principio di Selezione Clonale (Artificial Immune System)
    """

    def __init__(self, instance, pop_size=None, max_evals=20000):
        self.instance = instance
        
        # Popolazione Auto-Adattiva basata sulla grandezza dell'istanza! (25% del numero di nodi. Minimo 15 individui, Massimo 100.)
        if pop_size is None:
            self.pop_size = max(10, min(100, int(self.instance.num_nodes * 0.20)))
        else:
            self.pop_size = pop_size
            
        self.max_evals = max_evals
        self.evals = 0
        self.population = [] 
        self.memory = [] 
        self.best_cost_tracker = float('inf')
        self.convergence_eval = 0

    def is_valid(self, solution):
        for u, v in self.instance.edges:
            if solution[u] == 0 and solution[v] == 0:
                return False
        return True

    def calculate_cost(self, solution):
        return sum(self.instance.weights[i] * solution[i] for i in range(self.instance.num_nodes))

    def generate_greedy_solution(self, randomization_factor=0.2):
        """ Genera soluzione con K-Tournament e Redundancy Removal """

        sol = [0] * self.instance.num_nodes                                     #inizializzazione
        uncovered_degree = [len(adj) for adj in self.instance.adj_list]         #per ogni nod dice quanti archi non sono stati assegnati
        uncovered_edges_count = len(self.instance.edges)                        #numero totale di archi da dover ancora assengare
        
        active_nodes = {i for i in range(self.instance.num_nodes) if uncovered_degree[i] > 0}           #nodi da attivare
        
        while uncovered_edges_count > 0:
            
            candidates_list = list(active_nodes)
            k_tournament = min(5, len(candidates_list))
            tournament_pool = random.sample(candidates_list, k_tournament)      #preleva k elementi
            
            best_node = -1
            best_score = float('inf')
            
            for node in tournament_pool:
                noise = random.uniform(0, 0.001)
                score = (self.instance.weights[node] / uncovered_degree[node]) + noise
                if score < best_score:
                    best_score = score
                    best_node = node
                    
            sol[best_node] = 1                                                  #migliore del torneo è impostato a 1
            active_nodes.discard(best_node)                                     # Rimuoviamo il nodo scelto dai candidati
            
            #assegnazione ai vicini 
            for neighbor in self.instance.adj_list[best_node]:
                if sol[neighbor] == 0 and uncovered_degree[neighbor] > 0:
                    uncovered_degree[neighbor] -= 1
                    uncovered_edges_count -= 1

                    if uncovered_degree[neighbor] == 0:
                        active_nodes.discard(neighbor) # Se non ha più archi scoperti, lo ignoriamo
            uncovered_degree[best_node] = 0
            
        #Redundancy Removal Stocastico 
        nodes_in_sol = [i for i in range(self.instance.num_nodes) if sol[i] == 1]
        random.shuffle(nodes_in_sol) 
        
        #rimuovo un nodo se tutti i vicini sono attivi
        for node in nodes_in_sol:
            can_remove = True
            for neighbor in self.instance.adj_list[node]:
                if sol[neighbor] == 0:
                    can_remove = False
                    break
            if can_remove:
                sol[node] = 0 
                
        return sol

    def initialize_population(self):
        print("Generazione pop")
        for _ in range(self.pop_size):
            sol = self.generate_greedy_solution()
            cost = self.calculate_cost(sol)
            self.population.append({'solution': sol, 'cost': cost, 'affinity': 0})
            self.evals += 1
            
        self.population.sort(key=lambda x: x['cost'])                                       #popolazione ordinata in base al costo
        self.memory = self.population[:max(1, int(self.pop_size * 0.2))].copy()             #copia la top 20 %  delle soluzionio 

    def mutate_and_repair(self, solution, mutation_prob):
        mutated = solution.copy()                               #clonazione
        
        # mutazione
        for i in range(self.instance.num_nodes):
            if random.random() < mutation_prob:
                mutated[i] = 1 - mutated[i]
                
        uncovered_degree = [0] * self.instance.num_nodes
        uncovered_edges_count = 0
        
        # Tracciamo solo i nodi che hanno effettivamente archi scoperti
        active_nodes = set()
        for u, v in self.instance.edges:       
            if mutated[u] == 0 and mutated[v] == 0:
                uncovered_degree[u] += 1
                uncovered_degree[v] += 1
                uncovered_edges_count += 1
                active_nodes.add(u)
                active_nodes.add(v)
                
        while uncovered_edges_count > 0:
            best_node, best_score = -1, float('inf')
            
            
            for i in active_nodes:
                noise = random.uniform(0, 0.001)
                score = (self.instance.weights[i] / uncovered_degree[i]) + noise
                if score < best_score:
                    best_score, best_node = score, i
                    
            mutated[best_node] = 1
            active_nodes.discard(best_node)
            
            for neighbor in self.instance.adj_list[best_node]:
                if mutated[neighbor] == 0 and uncovered_degree[neighbor] > 0:
                    uncovered_degree[neighbor] -= 1
                    uncovered_edges_count -= 1
                    if uncovered_degree[neighbor] == 0:
                        active_nodes.discard(neighbor)
            uncovered_degree[best_node] = 0
                    
        #Redundancy Removal Stocastico 
        nodes_in_sol = [i for i in range(self.instance.num_nodes) if mutated[i] == 1]
        random.shuffle(nodes_in_sol)
        
        for node in nodes_in_sol:
            can_remove = True
            for neighbor in self.instance.adj_list[node]:
                if mutated[neighbor] == 0:
                    can_remove = False
                    break
            if can_remove:
                mutated[node] = 0
                
        return mutated

    def run(self):
        self.initialize_population()
        generation = 0
        beta_clones = 0.2 

        while self.evals < self.max_evals:
            generation += 1
            
            costs = [ab['cost'] for ab in self.population]
            min_cost, max_cost = min(costs), max(costs)
            
            for ab in self.population:
                if max_cost == min_cost:
                    ab['affinity'] = 1.0
                else:
                    ab['affinity'] = (max_cost - ab['cost']) / (max_cost - min_cost)

            new_population = []

            for ab in self.population:
                num_clones = max(1, int(self.pop_size * beta_clones * ab['affinity']))
                best_match = {'solution': ab['solution'].copy(), 'cost': ab['cost'], 'affinity': ab['affinity']}
                
                for _ in range(num_clones):
                    if self.evals >= self.max_evals: break
                    
                
                    mutation_prob = 0.05 + 0.45 * (1.0 - ab['affinity'])
                    
                    mutated_sol = self.mutate_and_repair(ab['solution'], mutation_prob)
                    cost = self.calculate_cost(mutated_sol)
                    self.evals += 1
                    
                    if cost < best_match['cost']:
                        best_match = {'solution': mutated_sol, 'cost': cost, 'affinity': 0}
                
                new_population.append(best_match)
                if self.evals >= self.max_evals: break

            memory_size = max(1, int(self.pop_size * 0.2))

            combined_memory = self.memory + new_population          #memoria composta dalla popolazione e le mutazioni
            combined_memory.sort(key=lambda x: x['cost'])
            
            unique_memory = []
            seen_signatures = set()

            for ind in combined_memory:
                sig = tuple(ind['solution']) 
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    unique_memory.append(ind)
                if len(unique_memory) == memory_size:
                    break
            self.memory = unique_memory


            #rimozione doppioni
            unique_pop = []
            seen_pop_sig = set()
            for ind in new_population:
                sig = tuple(ind['solution'])            #sig è una tupla della soluzione 
                if sig not in seen_pop_sig:
                    seen_pop_sig.add(sig)
                    unique_pop.append(ind)
                    
            #se ci sono meno elementi della pop_size genera altri
            while len(unique_pop) < self.pop_size and self.evals < self.max_evals:
                 sol = self.generate_greedy_solution()
                 unique_pop.append({'solution': sol, 'cost': self.calculate_cost(sol), 'affinity': 0})
                 self.evals += 1

            #rimpiazza i peggiori
            unique_pop.sort(key=lambda x: x['cost'])
            if self.evals < self.max_evals:
                num_to_replace = max(1, int(self.pop_size * 0.1))
                for i in range(len(unique_pop) - num_to_replace, len(unique_pop)):
                    sol = self.generate_greedy_solution()
                    unique_pop[i] = {'solution': sol, 'cost': self.calculate_cost(sol), 'affinity': 0}
                    self.evals += 1

            self.population = unique_pop
            
            if self.memory[0]['cost'] < self.best_cost_tracker:
                self.best_cost_tracker = self.memory[0]['cost']
                self.convergence_eval = self.evals
            
            if generation % 5 == 0:
                print(f"Gen {generation} | FE: {self.evals}/{self.max_evals} | Pop_Best: {self.population[0]['cost']} | MEMORY_BEST: {self.memory[0]['cost']}")

        print(f"\nRicerca completata! Miglior costo finale (Memory Cell): {self.memory[0]['cost']}")
        return self.memory[0]

if __name__ == "__main__":

    from collections import defaultdict
    
    cartella_istanze = "wvcp-instances" 
    
    BENCHMARK_TARGETS = {
        "vc_20_60": {"avg": 861.8, "eval": 7.7},
        "vc_20_120": {"avg": 1038.2, "eval": 5.2},
        "vc_25_150": {"avg": 1264.0, "eval": 21.0},
        "vc_100_500": {"avg": 4600.6, "eval": 703.0},
        "vc_100_2000": {"avg": 6051.9, "eval": 307.0},
        "vc_200_750": {"avg": 8274.5, "eval": 995.6},
        "vc_200_3000": {"avg": 11600.2, "eval": 690.6},
        "vc_800_10000": {"avg": 44397.8, "best": 44396.0, "eval": 4521.9}
    }
    
    print("\n" + "="*80)
    print(" AVVIO BENCHMARK GLOBALE")
    print("="*80)
    
    if not os.path.exists(cartella_istanze):
        print(f"\nERRORE CRITICO: La cartella '{cartella_istanze}' non esiste!")
        sys.exit(1)
        
    file_nella_cartella = [f for f in os.listdir(cartella_istanze) if f.endswith('.txt')]
    tutti_i_file = []
    
    for nome_file in file_nella_cartella:
        parti_nome = nome_file.replace('.txt', '').split('_')
        if len(parti_nome) >= 3:
            famiglia = f"{parti_nome[0]}_{parti_nome[1]}_{parti_nome[2]}"
            if famiglia in BENCHMARK_TARGETS:
                tutti_i_file.append(nome_file)
                
    tutti_i_file.sort()
    
    if not tutti_i_file:
        print(f"\n ERRORE CRITICO: Nessun file corrispondente ai target trovato in '{cartella_istanze}'!")
        sys.exit(1)
        
    print(f"[*] Trovate {len(tutti_i_file)} istanze valide. Inizio elaborazione...\n")
    
    risultati_raggruppati = defaultdict(lambda: {'costi': [], 'evals': []})

    for nome_file in tutti_i_file:
        percorso_completo = os.path.join(cartella_istanze, nome_file)
        
        parti_nome = nome_file.replace('.txt', '').split('_')
        famiglia = f"{parti_nome[0]}_{parti_nome[1]}_{parti_nome[2]}"
            
        print(f" Elaborazione: {nome_file:<20} ...", end=" ", flush=True)
        
        try:
            istanza = MWVCInstance(filepath=percorso_completo)
        except Exception as e:
            print(f"[-] Errore: {e}")
            continue
            
        start_time = time.time()
        
        algoritmo = Immune_Inspired(istanza, max_evals=20000)
        miglior_soluzione = algoritmo.run()
        
        elapsed_time = time.time() - start_time
        costo_istanza = miglior_soluzione['cost']
        eval_convergenza = algoritmo.convergence_eval
        
        risultati_raggruppati[famiglia]['costi'].append(costo_istanza)
        risultati_raggruppati[famiglia]['evals'].append(eval_convergenza)
        
        print(f"Fatto! Costo: {costo_istanza:.1f} | FE: {eval_convergenza} | Tempo: {elapsed_time:.2f}s")

    print("\n\n" + "#"*80)
    print(" REPORT DI VALUTAZIONE FINALE (Confronto con PBIG/ACO)")
    print("#"*80)
    
    for famiglia, dati in risultati_raggruppati.items():
        costi = dati['costi']
        evals = dati['evals']
        
        media_costo = sum(costi) / len(costi)
        best_costo = min(costi)
        media_eval = sum(evals) / len(evals)
        
        target = BENCHMARK_TARGETS.get(famiglia)
        
        print(f"\n--- FAMIGLIA: {famiglia} (Testate {len(costi)} istanze) ---")
        
        if target:
            diff_media = media_costo - target['avg']
            segno_media = "+" if diff_media > 0 else ""
            val_media = "OTTIMO/MIGLIORE" if diff_media <= 0 else "Da ottimizzare"
            
            print(f"  [COSTO] Media CLONALG: \t{media_costo:.2f}")
            print(f"  [COSTO] Media Target:  \t{target['avg']:.2f} \t[{segno_media}{diff_media:.2f} -> {val_media}]")
            
            diff_eval = media_eval - target['eval']
            segno_eval = "+" if diff_eval > 0 else ""
            val_eval = "PIÙ VELOCE" if diff_eval <= 0 else "Più lento"
            
            print(f"  [EVALS] Media CLONALG: \t{media_eval:.1f} FE")
            print(f"  [EVALS] Media Target:  \t{target['eval']:.1f} FE \t[{segno_eval}{diff_eval:.1f} -> {val_eval}]")
            
            if 'best' in target:
                diff_best = best_costo - target['best']
                segno_best = "+" if diff_best > 0 else ""
                val_best = "OTTIMO/MIGLIORE" if diff_best <= 0 else "Da ottimizzare"
                
                print(f"  [BEST ] Best CLONALG:  \t{best_costo:.2f}")
                print(f"  [BEST ] Best Target:   \t{target['best']:.2f} \t[{segno_best}{diff_best:.2f} -> {val_best}]")

    print("\n" + "="*80)
    print(" TEST MASSIVO COMPLETATO!")