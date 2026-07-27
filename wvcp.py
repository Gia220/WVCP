import random
import sys
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class WVCPInstance:


    def __init__(self, filepath=None, raw_data=None):
        self.num_nodes = 0
        self.weights = []
        self.adj_matrix = []
        self.edges = []
        self.adj_list = [] 
        
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

class Immune_Inspired_Algorithm:

    def __init__(self, instance, pop_size=None, max_evals=20000):
        self.instance = instance
        
        # Popolazione Auto-Adattiva basata sulla grandezza dell'istanza! (20% del numero di nodi. Minimo 10 individui, Massimo 100.)
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
        self.history = []  # Registro per il grafico di convergenza

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
        uncovered_degree = [len(adj) for adj in self.instance.adj_list]         #numero di archi per ogni nodi da assegnare
        uncovered_edges_count = len(self.instance.edges)                        #numero totale di archi da dover ancora assengare
        
        active_nodes = {i for i in range(self.instance.num_nodes) if uncovered_degree[i] > 0}           #nodi con almeno un grado di libertà
        
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
            active_nodes.discard(best_node)                                     #Rimuoviamo il nodo scelto dai candidati
            
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

    def initialize_population2(self):
        print("Generazione popolazione iniziale.")
        for _ in range(self.pop_size):
            sol = self.generate_greedy_solution()
            cost = self.calculate_cost(sol)
            self.evals += 1
            self.population.append({'solution': sol, 'cost': cost, 'affinity': 0})
            
        self.population.sort(key=lambda x: x['cost'])                                       #popolazione ordinata in base al costo
        self.memory = self.population[:max(1, int(self.pop_size * 0.2))].copy()             #copia la top 20 %  delle soluzionio 

    def initialize_population(self):
        print("Generazione popolazione iniziale.")
        for _ in range(self.pop_size):
            sol = self.generate_greedy_solution()
            cost = self.calculate_cost(sol)
            self.evals += 1
            self.population.append({'solution': sol, 'cost': cost, 'affinity': 0})
            

            if self.evals == 1:
                self.best_cost_tracker = cost
                self.history.append((self.evals, self.best_cost_tracker))
            elif cost < self.best_cost_tracker:
                self.best_cost_tracker = cost
                self.history.append((self.evals, self.best_cost_tracker))
            
        self.population.sort(key=lambda x: x['cost'])                                       # Popolazione ordinata in base al costo
        self.memory = self.population[:max(1, int(self.pop_size * 0.2))].copy()             # Copia la top 20% delle soluzioni

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
            
            #calcolo affinità
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
            
            #rimozione doppiono della memory
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

            #rimozione doppioni dalla popolazione
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


            unique_pop.sort(key=lambda x: x['cost']) 
            
            if self.evals < self.max_evals:
                #  MEMORY REINTRODUCTION
                # Ogni 10 generazioni, i campioni in memoria rimpiazzano gli anticorpi peggiori
                if generation % 10 == 0 and len(self.memory) > 0:
                    num_to_inject = max(1, int(self.pop_size * 0.1))
                    num_to_inject = min(num_to_inject, len(self.memory))
                    
                    for i in range(num_to_inject):
                        indice_da_sostituire = len(unique_pop) - 1 - i
                        anticorpo_memoria = {
                            'solution': list(self.memory[i]['solution']), # deep copy
                            'cost': self.memory[i]['cost'],
                            'affinity': 0 
                        }
                        unique_pop[indice_da_sostituire] = anticorpo_memoria

                # RECEPTOR EDITING 
                # Ogni 5 generazioni, i peggiori vengono sostituiti con soluzioni casuali nuove
                elif generation % 5 == 0:
                    num_to_replace = max(1, int(self.pop_size * 0.1))
                    for i in range(len(unique_pop) - num_to_replace, len(unique_pop)):
                        sol = self.generate_greedy_solution()
                        unique_pop[i] = {'solution': sol, 'cost': self.calculate_cost(sol), 'affinity': 0}
                        self.evals += 1

            self.population = unique_pop
            
            if self.memory[0]['cost'] < self.best_cost_tracker:
                self.best_cost_tracker = self.memory[0]['cost']
                self.convergence_eval = self.evals
            
            self.history.append((self.evals, self.best_cost_tracker))
            
            if False:#generation % 5 == 0:
                print(f"Gen {generation} | FE: {self.evals}/{self.max_evals} | Pop_Best: {self.population[0]['cost']} | MEMORY_BEST: {self.memory[0]['cost']}")

        print(f"\nRicerca completata! Miglior costo finale (Memory Cell): {self.memory[0]['cost']}")
        return self.memory[0]

# COSTANTI GLOBALI
CARTELLA_ISTANZE = "wvcp-instances" 
CARTELLA_GRAFICI = "grafici"
TEST_VELOCE = True 
FAMIGLIE_TEST = ["vc_800_10000" ] 


def benchmark_targets():
    """Restituisce il dizionario dei target, filtrandolo se in TEST_VELOCE."""
    targets = {
        "vc_20_60": {"avg": 861.8, "eval": 7.7},
        "vc_20_120": {"avg": 1038.2, "eval": 5.2},
        "vc_25_150": {"avg": 1264.0, "eval": 21.0},
        "vc_100_500": {"avg": 4600.6, "eval": 703.0},
        "vc_100_2000": {"avg": 6051.9, "eval": 307.0},
        "vc_200_750": {"avg": 8274.5, "eval": 995.6},
        "vc_200_3000": {"avg": 11600.2, "eval": 690.6},
        "vc_800_10000": {"avg": 44397.8, "best": 44396.0, "eval": 4521.9}
    }
    
    if TEST_VELOCE:
        targets = {k: v for k, v in targets.items() if k in FAMIGLIE_TEST}
        
    return targets

def istanze_valide(cartella_istanze, targets):
    """Cerca e restituisce la lista dei file validi per il benchmark."""

    if not os.path.exists(cartella_istanze):
        print(f"\n ERRORE: La cartella '{cartella_istanze}' non esiste!")
        sys.exit(1)
        
    tutti_i_file = []
    for nome_file in os.listdir(cartella_istanze):
        if nome_file.endswith('.txt'):
            parti_nome = nome_file.replace('.txt', '').split('_')
            if len(parti_nome) >= 3:
                famiglia = f"{parti_nome[0]}_{parti_nome[1]}_{parti_nome[2]}"
                if famiglia in targets:
                    tutti_i_file.append(nome_file)
                    
    tutti_i_file.sort()
    
    if not tutti_i_file:
        print(f"\n ERRORE: Nessun file corrispondente ai target trovato in '{cartella_istanze}'!")
        sys.exit(1)
        
    return tutti_i_file

def elaborazione_totale2(lista_file, cartella_istanze):
    """Esegue l'algoritmo su tutte le istanze e raccoglie i dati."""
    print(f" Trovate {len(lista_file)} istanze valide. Inizio elaborazione...\n")
    
    # MODIFICA: Aggiunto 'all_histories' per salvare i grafici di tutte le 10 run
    risultati = defaultdict(lambda: {
        'costi': [], 
        'evals': [], 
        'best_history': [], 
        'best_storico_costo': float('inf'),
        'all_histories': [] 
    })

    for nome_file in lista_file:
        percorso_completo = os.path.join(cartella_istanze, nome_file)
        famiglia = "_".join(nome_file.replace('.txt', '').split('_')[:3])
            
        print(f" Elaborazione: {nome_file:<20} ...", end=" ", flush=True)
        
        try:
            istanza = WVCPInstance(filepath=percorso_completo)
        except Exception as e:
            print(f"[-] Errore caricamento istanza: {e}")
            continue
            
        start_time = time.time()
        algoritmo = Immune_Inspired_Algorithm(istanza, max_evals=20000) 
        miglior_soluzione = algoritmo.run()
        elapsed_time = time.time() - start_time
        
        costo = miglior_soluzione['cost']
        evals = algoritmo.convergence_eval
        
        risultati[famiglia]['costi'].append(costo)
        risultati[famiglia]['evals'].append(evals)
        
        # MODIFICA: Salviamo la storia di QUESTA singola run nell'archivio totale
        risultati[famiglia]['all_histories'].append(algoritmo.history.copy())
        
        if costo < risultati[famiglia]['best_storico_costo']:
            risultati[famiglia]['best_storico_costo'] = costo
            risultati[famiglia]['best_history'] = algoritmo.history.copy()
        
        print(f"Fatto! Costo: {costo:.1f} | FE: {evals} | Tempo: {elapsed_time:.2f}s")
        
    return risultati

def elaborazione_totale(lista_file, cartella_istanze):
    """Esegue l'algoritmo su tutte le istanze e raccoglie i dati."""
    print(f" Trovate {len(lista_file)} istanze valide. Inizio elaborazione...\n")
    
    risultati = defaultdict(lambda: {
        'costi': [], 
        'evals': [], 
        'best_history': [], 
        'best_storico_costo': float('inf'),
        'all_histories': [] 
    })

    for nome_file in lista_file:
        percorso_completo = os.path.join(cartella_istanze, nome_file)
        famiglia = "_".join(nome_file.replace('.txt', '').split('_')[:3])
            
        try:
            istanza = WVCPInstance(filepath=percorso_completo)
        except Exception as e:
            print(f" Errore caricamento istanza: {e}")
            continue
            
      
        # Se è l'istanza grande (LPI), fai 10 run sullo stesso file. 
        numero_run = 10 if famiglia == "vc_800_10000" else 1

        for run_idx in range(numero_run):
            if numero_run > 1:
                print(f" Elaborazione: {nome_file:<20} (Run {run_idx+1}/{numero_run})...", end=" ", flush=True)
            else:
                print(f" Elaborazione: {nome_file:<20} ...", end=" ", flush=True)
            
            start_time = time.time()
            # Passiamo l'istanza caricata all'algoritmo
            algoritmo = Immune_Inspired_Algorithm(istanza, max_evals=20000) 
            miglior_soluzione = algoritmo.run()
            elapsed_time = time.time() - start_time
            
            costo = miglior_soluzione['cost']
            evals = algoritmo.convergence_eval
            
            # Salviamo i risultati. Essendo la stessa 'famiglia', si accumuleranno tutti e 10
            risultati[famiglia]['costi'].append(costo)
            risultati[famiglia]['evals'].append(evals)
            
            # Salviamo la storia di convergenza per il grafico
            risultati[famiglia]['all_histories'].append(algoritmo.history.copy())
            
            if costo < risultati[famiglia]['best_storico_costo']:
                risultati[famiglia]['best_storico_costo'] = costo
                risultati[famiglia]['best_history'] = algoritmo.history.copy()
            
            print(f"Fatto! Costo: {costo:.1f} | FE: {evals} | Tempo: {elapsed_time:.2f}s")
        
    return risultati

def report_testuale(risultati, targets):

    print("\n\n" + "#"*80)
    print(" REPORT DI VALUTAZIONE FINALE (Confronto con PBIG/ACO)")
    print("#"*80)
    
    for famiglia, dati in risultati.items():
        media_costo = sum(dati['costi']) / len(dati['costi'])
        media_eval = sum(dati['evals']) / len(dati['evals'])
        target = targets.get(famiglia)
        
        print(f"\n--- FAMIGLIA: {famiglia} (Testate {len(dati['costi'])} istanze) ---")
        if target:
            diff_media = media_costo - target['avg']
            val_media = "OTTIMO/MIGLIORE" if diff_media <= 0 else "Da ottimizzare"
            print(f"  [COSTO] Media IA: \t{media_costo:.2f}")
            print(f"  [COSTO] Media Target:  \t{target['avg']:.2f} \t[{'+' if diff_media > 0 else ''}{diff_media:.2f} -> {val_media}]")
            
            diff_eval = media_eval - target['eval']
            val_eval = "PIÙ VELOCE" if diff_eval <= 0 else "Più lento"
            print(f"  [EVALS] Media IA: \t{media_eval:.1f} FE")
            print(f"  [EVALS] Media Target:  \t{target['eval']:.1f} FE \t[{'+' if diff_eval > 0 else ''}{diff_eval:.1f} -> {val_eval}]")

    print("\n" + "="*80)
    print(" TEST MASSIVO COMPLETATO!")

def grafici_convergenza(risultati, cartella_grafici):
    """Genera i plot con tutte le run, la media e la run migliore."""
    print("\n Generazione dei grafici di convergenza in corso...")
    os.makedirs(cartella_grafici, exist_ok=True)

    for famiglia, dati in risultati.items():
        tutte_le_storie = dati['all_histories']

        if not tutte_le_storie:
            continue

        plt.figure(figsize=(10, 6)) # Leggermente allargato per comodità

        #Trova la lunghezza massima dell'asse X (FE) per poter calcolare la media
        max_fe = 0
        for storia in tutte_le_storie:
            if storia:
                max_fe = max(max_fe, storia[-1][0])
                
        if max_fe == 0:
            continue

        # Creiamo un asse X comune di 500 punti per allineare matematicamente le varie run
        x_common = np.linspace(0, max_fe, 500)
        y_somma = np.zeros(500)
        run_valide = 0

        # Prendiamo 10 colori distinti dalla palette 'tab10'
        colori_run = plt.cm.tab10(np.linspace(0, 1, 10))

        #DISEGNA LE SINGOLE RUN
        for i, storia in enumerate(tutte_le_storie):
            if not storia:
                continue
            xs = [p[0] for p in storia]
            ys = [p[1] for p in storia]
            
            # Etichetta solo per la prima run, per non intaccare la legenda 10 volte
            label_singola = 'Singole Run' if i == 0 else ""
            
            # Assegniamo un colore diverso ad ogni run usando l'indice 'i'
            colore_corrente = colori_run[i % 10]
            
            # Traccia la singola run
            plt.plot(xs, ys, color=colore_corrente, alpha=0.75, linewidth=1.5, label=label_singola)
            
            # Interpola i valori y su x_common per poterli sommare
            y_interp = np.interp(x_common, xs, ys)
            y_somma += y_interp
            run_valide += 1

        #CALCOLA E DISEGNA LA MEDIA GLOBALE
        if run_valide > 0:
            y_media = y_somma / run_valide
            # Usiamo un rosso scuro (#d62728) tratteggiato e più spesso per far risaltare la media
            plt.plot(x_common, y_media, color='#d62728', linestyle='--', linewidth=3, label='Convergenza Media')


        # Formattazione estetica
        plt.title(f'Convergenza Globale - Famiglia: {famiglia}', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Fitness Evaluations (FE)', fontsize=11, labelpad=10)
        plt.ylabel('Costo del Vertex Cover', fontsize=11, labelpad=10)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(frameon=True, facecolor='white', edgecolor='none', shadow=True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        
        plt.savefig(os.path.join(cartella_grafici, f'convergenza_{famiglia}.png'), dpi=300)
        plt.close() 
            
    print(f"[+] Generati {len(risultati)} grafici di convergenza dettagliati in '{cartella_grafici}'.")

def genera_grafici_riassuntivi(risultati, targets, cartella_grafici):
    """Genera i grafici a barre riassuntivi per Costi e FE."""
    print("[*] Generazione dei grafici a barre riassuntivi in corso...")
    
    nomi_famiglie, costi_IA, costi_target, fe_IA, fe_target = [], [], [], [], []

    for famiglia, dati in risultati.items():
        nomi_famiglie.append(famiglia.replace('vc_', ''))
        costi_IA.append(sum(dati['costi']) / len(dati['costi']))
        fe_IA.append(sum(dati['evals']) / len(dati['evals']))
        costi_target.append(targets[famiglia]['avg'])
        fe_target.append(targets[famiglia]['eval'])

    x_pos = np.arange(len(nomi_famiglie))
    width = 0.35 

    # --- GRAFICO A BARRE: COSTI ---
    plt.figure(figsize=(12, 6))
    plt.bar(x_pos - width/2, costi_IA, width, label='IA', color='#1f77b4')
    plt.bar(x_pos + width/2, costi_target, width, label='Target (PBIG)', color='#ff7f0e')

    plt.ylabel('Costo Medio', fontsize=12)
    plt.title('Confronto Costo Medio', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, nomi_famiglie, rotation=45, ha='right')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylim(0, max(max(costi_IA), max(costi_target)) * 1.1) 
    plt.tight_layout()
    plt.savefig(os.path.join(cartella_grafici, 'riassunto_costi.png'), dpi=300)
    plt.close()

    # --- GRAFICO A BARRE: VALUTAZIONI (FE) ---
    plt.figure(figsize=(10, 6))
    barre_ia = plt.bar(x_pos - width/2, fe_IA, width, label='IA', color='#2ca02c')
    barre_target = plt.bar(x_pos + width/2, fe_target, width, label='Target (PBIG)', color='#d62728')

    plt.gca().bar_label(barre_ia, padding=3, fmt='%.1f', fontsize=10, fontweight='bold', color='#2ca02c')
    plt.gca().bar_label(barre_target, padding=3, fmt='%.1f', fontsize=10, fontweight='bold', color='#d62728')

    plt.ylabel('Fitness Evaluations Medie', fontsize=12)
    plt.title('Confronto Valutazioni (FE)', fontsize=15, fontweight='bold', pad=20)
    plt.xticks(x_pos, nomi_famiglie, rotation=45, ha='right')
    plt.legend(fontsize=11)
    plt.yscale('log')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(cartella_grafici, 'riassunto_fe.png'), dpi=300)
    plt.close()

    print(f"[+] Generati grafici riassuntivi a barre in '{cartella_grafici}'.")
    print("\n" + "="*80)
    print(" TUTTE LE OPERAZIONI SONO CONCLUSE CON SUCCESSO! ")
    print("="*80)


def main():
    print("\n" + "="*80)
    print(" AVVIO BENCHMARK GLOBALE " + ("[MODALITÀ TEST VELOCE]" if TEST_VELOCE else "[FULL RUN]"))
    print("="*80)

    targets = benchmark_targets()
    
    # cerca i file dalla cartella
    file_da_elaborare = istanze_valide(CARTELLA_ISTANZE, targets)
    
    # calcolo per tutte le instanze
    risultati_finali = elaborazione_totale(file_da_elaborare, CARTELLA_ISTANZE)
    
    report_testuale(risultati_finali, targets)
    

    grafici_convergenza(risultati_finali, CARTELLA_GRAFICI)
    genera_grafici_riassuntivi(risultati_finali, targets, CARTELLA_GRAFICI)

if __name__ == "__main__":
     main()