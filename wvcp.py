

   import random

class MWVCInstance:
    """
    Classe che rappresenta un'istanza del problema Minimum Weight Vertex Cover.
    """
    def __init__(self, filepath=None, raw_data=None):
        self.num_nodes = 0
        self.weights = []
        self.adj_matrix = []
        self.edges = []
        self.adj_list = [] # NUOVO: Lista di adiacenza per calcoli super-veloci
        
        # Supportiamo la lettura diretta dal file!
        if filepath:
            with open(filepath, 'r') as f:
                self._parse_data(f.read())
        elif raw_data:
            self._parse_data(raw_data)
        else:
            raise ValueError("Devi fornire un filepath o raw_data.")

    def _parse_data(self, raw_data):
        # Rimuoviamo righe vuote e dividiamo per riga
        lines = [line.strip() for line in raw_data.strip().split('\n') if line.strip()]
        
        # 1. Numero di vertici (prima riga)
        self.num_nodes = int(lines[0])
        
        # 2. Pesi dei vertici (seconda riga)
        self.weights = [float(w) for w in lines[1].split()]
        
        # 3. Matrice di adiacenza (dalla terza riga in poi)
        for i in range(2, 2 + self.num_nodes):
            row = [int(val) for val in lines[i].split()]
            self.adj_matrix.append(row)
            
        # 4. Estrazione della lista degli archi e lista di adiacenza
        self.adj_list = [[] for _ in range(self.num_nodes)] # Inizializza liste vuote per ogni nodo
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes): # i+1 per evitare duplicati
                if self.adj_matrix[i][j] == 1:
                    self.edges.append((i, j))
                    self.adj_list[i].append(j) # Aggiunge j come vicino di i
                    self.adj_list[j].append(i) # Aggiunge i come vicino di j
                    
        print(f"[+] Istanza caricata con successo: {self.num_nodes} nodi, {len(self.edges)} archi.")

class ClonalSelection:
    """
    Algoritmo basato sul Principio di Selezione Clonale (Artificial Immune System)
    """
    def __init__(self, instance, pop_size=50, max_evals=20000):
        self.instance = instance
        self.pop_size = pop_size
        self.max_evals = max_evals
        self.evals = 0
        self.population = [] 

    def is_valid(self, solution):
        """ Verifica rapida se la soluzione copre tutti gli archi """
        for u, v in self.instance.edges:
            if solution[u] == 0 and solution[v] == 0:
                return False
        return True

    def calculate_cost(self, solution):
        """ Calcola il peso totale dei vertici selezionati """
        return sum(self.instance.weights[i] * solution[i] for i in range(self.instance.num_nodes))

    def generate_greedy_solution(self, randomization_factor=0.2):
        """ Genera una soluzione usando un approccio Greedy SUPER OTTIMIZZATO """
        sol = [0] * self.instance.num_nodes
        
        # Invece di set lenti, tracciamo quanti archi "scoperti" tocca ogni nodo
        uncovered_degree = [len(adj) for adj in self.instance.adj_list]
        uncovered_edges_count = len(self.instance.edges)
        
        while uncovered_edges_count > 0:
            candidates = []
            for i in range(self.instance.num_nodes):
                if sol[i] == 0 and uncovered_degree[i] > 0:
                    score = self.instance.weights[i] / uncovered_degree[i]
                    candidates.append((score, i))
            
            candidates.sort(key=lambda x: x[0])
            top_k = max(1, int(len(candidates) * randomization_factor))
            chosen_node = random.choice(candidates[:top_k])[1]
            
            sol[chosen_node] = 1
            
            # Aggiornamento fulmineo dei gradi: notifichiamo ai vicini che questo arco è coperto
            for neighbor in self.instance.adj_list[chosen_node]:
                if sol[neighbor] == 0:
                    uncovered_degree[neighbor] -= 1
                    uncovered_edges_count -= 1
            uncovered_degree[chosen_node] = 0 # Questo nodo non coprirà più nuovi archi
            
        # Redundancy removal (Fase di pulizia OTTIMIZZATA)
        nodes_in_sol = [i for i in range(self.instance.num_nodes) if sol[i] == 1]
        nodes_in_sol.sort(key=lambda i: self.instance.weights[i], reverse=True)
        
        for node in nodes_in_sol:
            can_remove = True
            # Un nodo è inutile SOLO SE tutti i suoi vicini sono già accesi nella soluzione
            for neighbor in self.instance.adj_list[node]:
                if sol[neighbor] == 0:
                    can_remove = False
                    break
            
            if can_remove:
                sol[node] = 0 
                
        return sol

    def initialize_population(self):
        """ Inizializza la generazione zero """
        print("[*] Inizializzazione della popolazione intelligente in corso...")
        for _ in range(self.pop_size):
            sol = self.generate_greedy_solution()
            cost = self.calculate_cost(sol)
            self.population.append({'solution': sol, 'cost': cost})
            self.evals += 1
        
        self.population.sort(key=lambda x: x['cost'])
        print(f"[+] Popolazione creata! Il miglior costo di partenza è: {self.population[0]['cost']}")

    def mutate_and_repair(self, solution, mutation_prob):
        """ Iper-mutazione con riparazione greedy OTTIMIZZATA """
        mutated = solution.copy()

        # 1. Flip dei bit (Mutazione vera e propria)
        for i in range(self.instance.num_nodes):
            if random.random() < mutation_prob:
                mutated[i] = 1 - mutated[i]
                
        # 2. Riparazione ultra-veloce
        uncovered_degree = [0] * self.instance.num_nodes
        uncovered_edges_count = 0
        
        # Identifichiamo subito gli archi rimasti scoperti
        for u, v in self.instance.edges:
            if mutated[u] == 0 and mutated[v] == 0:
                uncovered_degree[u] += 1
                uncovered_degree[v] += 1
                uncovered_edges_count += 1
                
        while uncovered_edges_count > 0:
            best_node = -1
            best_score = float('inf')
            
            for i in range(self.instance.num_nodes):
                if mutated[i] == 0 and uncovered_degree[i] > 0:
                    score = self.instance.weights[i] / uncovered_degree[i]
                    if score < best_score:
                        best_score = score
                        best_node = i
                        
            mutated[best_node] = 1
            
            # Aggiorniamo a cascata
            for neighbor in self.instance.adj_list[best_node]:
                if mutated[neighbor] == 0 and uncovered_degree[neighbor] > 0:
                    uncovered_degree[neighbor] -= 1
                    uncovered_edges_count -= 1
            uncovered_degree[best_node] = 0
                    
        # 3. Rimozione ridondanze rapida
        nodes_in_sol = [i for i in range(self.instance.num_nodes) if mutated[i] == 1]
        nodes_in_sol.sort(key=lambda i: self.instance.weights[i], reverse=True)
        
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

        while self.evals < self.max_evals:
            generation += 1
            clones_pool = []

            # --- 1. CLONAZIONE ---
            # Calcoliamo i cloni in base al rank (indice). Migliore = indice 0
            for rank, antibody in enumerate(self.population):
                # Il migliore genera più cloni. Usiamo una formula proporzionale.
                num_clones = max(1, int((self.pop_size * 0.5) / (rank + 1)))
                for _ in range(num_clones):
                    clones_pool.append({'solution': antibody['solution'].copy(), 'rank': rank})

            # --- 2. IPER-MUTAZIONE ---
            mutated_clones = []
            for clone in clones_pool:
                rank = clone['rank']
                # Tasso dinamico: il migliore (rank 0) muta del 5%, l'ultimo muta del 50%
                mutation_prob = 0.05 + 0.45 * (rank / self.pop_size)
                
                mutated_sol = self.mutate_and_repair(clone['solution'], mutation_prob)
                cost = self.calculate_cost(mutated_sol)
                self.evals += 1
                
                mutated_clones.append({'solution': mutated_sol, 'cost': cost})
                
                if self.evals >= self.max_evals:
                    break

            # --- 3. SELEZIONE (Sopravvivenza) ---
            # Uniamo genitori e cloni, ordiniamo e scartiamo i peggiori
            combined = self.population + mutated_clones
            combined.sort(key=lambda x: x['cost'])
            
            # Manteniamo la diversità genetica rimuovendo i duplicati esatti
            unique_population = []
            seen_costs = set()
            for ind in combined:
                if ind['cost'] not in seen_costs:
                    seen_costs.add(ind['cost'])
                    unique_population.append(ind)
                if len(unique_population) == self.pop_size:
                    break
            
            # Se scartando i duplicati la popolazione si è svuotata troppo, riempiamo
            while len(unique_population) < self.pop_size and self.evals < self.max_evals:
                 sol = self.generate_greedy_solution(0.5)
                 unique_population.append({'solution': sol, 'cost': self.calculate_cost(sol)})
                 self.evals += 1
            
            self.population = unique_population

            # --- 4. RECEPTOR EDITING (Ricambio Generazionale) ---
            # Sostituiamo il peggiore 10% con anticorpi totalmente nuovi
            if self.evals < self.max_evals:
                num_to_replace = max(1, int(self.pop_size * 0.1))
                for i in range(self.pop_size - num_to_replace, self.pop_size):
                    sol = self.generate_greedy_solution(0.4)
                    self.population[i] = {'solution': sol, 'cost': self.calculate_cost(sol)}
                    self.evals += 1

            if generation % 5 == 0:
                print(f"Generazione {generation} | Valutazioni: {self.evals}/{self.max_evals} | Miglior Costo: {self.population[0]['cost']}")
                
        print(f"\n[!] Ricerca completata! Miglior costo finale: {self.population[0]['cost']}")
        return self.population[0]

# --- MAIN BLOCK PER TESTARE ---
if __name__ == "__main__":
    import sys
    file_path = "wvcp-instances/wvcp-instances/vc_800_10000.txt"
    
    try:
        # Tenta di leggere dal file reale sul tuo PC
        istanza = MWVCInstance(filepath=file_path)
    except FileNotFoundError:
        print(f"[-] ERRORE CRITICO: File '{file_path}' non trovato. Esecuzione interrotta.")
        sys.exit(1)
        
    # Inizializziamo l'algoritmo (ho abbassato max_evals a 5000 per un test veloce)
    algoritmo = ClonalSelection(istanza, pop_size=30, max_evals=5000)
    
    # Lanciamo il ciclo di ottimizzazione!
    miglior_soluzione = algoritmo.run()
    
    print("\n--- RISULTATO FINALE ---")
    print(f"Vettore soluzione: {miglior_soluzione['solution']}")
    print(f"La soluzione è valida? {algoritmo.is_valid(miglior_soluzione['solution'])}")

# --- MAIN BLOCK PER TESTARE ---
if __name__ == "__main__":
    import sys
    file_path = "wvcp-instances/wvcp-instances/vc_800_10000.txt"
    
    try:
        # Tenta di leggere dal file reale sul tuo PC
        istanza = MWVCInstance(filepath=file_path)
    except FileNotFoundError:
        print(f"[-] ERRORE CRITICO: File '{file_path}' non trovato. Esecuzione interrotta.")
        sys.exit(1)
        
    # Inizializziamo l'algoritmo (ho abbassato max_evals a 5000 per un test veloce)
    algoritmo = ClonalSelection(istanza, pop_size=30, max_evals=5000)
    
    # Lanciamo il ciclo di ottimizzazione!
    miglior_soluzione = algoritmo.run()
    
    print("\n--- RISULTATO FINALE ---")
    print(f"Vettore soluzione: {miglior_soluzione['solution']}")
    print(f"La soluzione è valida? {algoritmo.is_valid(miglior_soluzione['solution'])}")

# --- MAIN BLOCK: TEST BENCH AUTOMATIZZATO ---
if __name__ == "__main__":
    import sys
    import os
    
    # 1. Configurazione del percorso
    # ATTENZIONE: Se la tua cartella è doppiamente annidata, cambia in "wvcp-instances/wvcp-instances"
    cartella_istanze = "wvcp-instances/wvcp-instances" 
    base_nome_file = "vc_100_2000_{:02d}.txt" # Il {:02d} formatta i numeri in 01, 02, 03...
    
    costo_totale = 0
    istanze_calcolate = 0
    risultati_dettagliati = []
    
    print("\n" + "="*50)
    print(" AVVIO BENCHMARK: CLASSE SPI (20 nodi, 60 archi)")
    print("="*50)
    
    # 2. Ciclo for per scansionare le 10 istanze (da 1 a 10 compresi)
    for i in range(1, 11):
        nome_file = base_nome_file.format(i)
        percorso_completo = os.path.join(cartella_istanze, nome_file)
        
        print(f"\n[*] Analisi in corso sull'istanza: {nome_file} ...")
        
        try:
            # Carichiamo il file
            istanza = MWVCInstance(filepath=percorso_completo)
        except FileNotFoundError:
            print(f"[-] ATTENZIONE: File '{percorso_completo}' non trovato. Salto all'istanza successiva.")
            continue # Salta al prossimo giro del ciclo for
            
        # 3. Avvio dell'Algoritmo (max_evals = 20000 come da paper scientifico)
        # N.B. Ho disattivato i print interni dell'algoritmo per non intasare lo schermo
        algoritmo = ClonalSelection(istanza, pop_size=50, max_evals=20000)
        
        # Salviamo l'output originale di print per silenziarlo temporaneamente se vuoi un output pulito
        # (Se vuoi vedere tutte le generazioni, puoi lasciare così)
        miglior_soluzione = algoritmo.run()
        
        costo_istanza = miglior_soluzione['cost']
        risultati_dettagliati.append((nome_file, costo_istanza))
        costo_totale += costo_istanza
        istanze_calcolate += 1

    # 4. Report Finale
    print("\n" + "#"*50)
    print(" REPORT FINALE BENCHMARK")
    print("#"*50)
    
    if istanze_calcolate > 0:
        for nome, costo in risultati_dettagliati:
            print(f" -> {nome}: \tCosto Ottenuto = {costo}")
            
        media_algoritmo = costo_totale / istanze_calcolate
        print("-" * 50)
        print(f" MEDIA CALCOLATA SU {istanze_calcolate} ISTANZE: {media_algoritmo:.2f}")
        print(f" BENCHMARK DA BATTERE (Tabella 1 - ACO): 861.80")
        print("#"*50)
        
        if media_algoritmo <= 861.8:
            print("\n[VITTORIA!] Hai eguagliato o battuto lo stato dell'arte (ACO)!")
        else:
            print("\n[DA MIGLIORARE] L'algoritmo funziona, ma dobbiamo ottimizzare mutazione e parametri.")
    else:
        print("\n[-] ERRORE CRITICO: Nessuna istanza è stata caricata. Controlla il nome della cartella!")
        sys.exit(1)