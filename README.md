# 🏆 Challenge Ramadan - SupNum (S3C'1447)
## Solveur RCPSP Avancé

Bienvenue dans notre application de l'état de l'art pour la résolution du problème de planification de projet sous contraintes de ressources (**RCPSP** - Resource-Constrained Project Scheduling Problem). 
Ce projet a été fièrement développé en **Python** dans le cadre du défi de programmation de notre université, **SupNum**.

### 🎯 Objectif du Projet
L'objectif principal de cette application est de trouver le planning optimal ou quasi-optimal minimisant la durée totale du projet (**Makespan / Cmax**) tout en respectant les contraintes complexes de précédence entre les tâches et les limites strictes des ressources.
Le code a été conçu et optimisé spécifiquement pour exceller sur les instances complexes de **`j60.sm`** (qui constituent le cœur de ce défi), tout en ignorant les instances plus simples ou hors-sujet.

### 🧠 Algorithmes Utilisés (L'Intelligence Artificielle de l'App)
Notre solveur hybride ne se contente pas d'une approche aléatoire, il combine plusieurs techniques de pointe :

1. **SSGS (Serial Schedule Generation Scheme)** : 
   C'est le constructeur de base. Il prend une liste de tâches priorisées et les place dans le temps le plus tôt possible, sans jamais violer les contraintes de ressources (par exemple: ne pas dépasser le nombre d'ouvriers disponibles) ni les contraintes de précédence (une tâche B ne peut commencer que si A est terminée).

2. **FBI (Forward-Backward Improvement)** : 
   Une méta-heuristique d'intensification extrêmement puissante. Elle prend un planning existant et "pousse" toutes les tâches le plus tard possible (Forward), puis les "écrase" à nouveau le plus tôt possible (Backward). Ce mouvement tactique découvre souvent des espaces vides cachés dans le planning, permettant à d'autres tâches d'être insérées et de réduire drastiquement la durée totale du projet.

3. **Scatter Search & Path Relinking** : 
   Pour surmonter les obstacles des instances difficiles (`j60.sm`), cette méthode prend un ensemble de nos meilleures solutions trouvées (les "élites"), et trace un chemin entre elles. En naviguant d'une bonne solution à une autre, l'algorithme "répare" et "optimise" les erreurs, garantissant une convergence vers l'optimum mondial, souvent en quelques secondes !

### ⚙️ Instructions d'Exécution & d'Utilisation

L'application est fournie avec une interface graphique (GUI) élégante et complète. L'utilisateur doit posséder les dossiers de test (ex: `j60.sm`) sur sa machine locale pour l'analyse.

1. **Prérequis** : 
   Assurez-vous d'avoir Python 3.8+ installé sur votre machine.
   Les dossiers de test (comme `j60.sm`) ne sont pas inclus dans ce dépôt pour le garder léger, vous devez les télécharger localement.

2. **Installation des dépendances** :
   ```bash
   pip install matplotlib pandas openpyxl python-docx
   ```

3. **Lancement de l'application** :
   Ouvrez un terminal dans le dossier du projet et exécutez le point d'entrée :
   ```bash
   python main.py
   ```

4. **Démarche sur l'Interface (GUI)** :
   - Cliquez sur le bouton **"Browse"** pour charger le dossier contenant vos instances de test (ex: le dossier des fichiers `.sm` de la catégorie `j60`).
   - Le menu de configuration vous permet de définir le **"Time Budget" (Temps alloué en secondes)**. C'est crucial : plus vous donnez de temps à l'algorithme, plus ses résultats s'affinent.
   - Sélectionnez un fichier dans la liste et cliquez sur **"Run Selected"**.
   - Le diagramme de Gantt interactif se mettra à jour en temps réel.
   - Accédez à l'onglet "Results" pour voir l'écart (Gap) par rapport à la Borne Inférieure (Lower Bound) et exporter vos résultats vers Word ou Excel.

---
*Ce dépôt ne contient intentionnellement que le cœur de l'intelligence artificielle (fichiers `.py`) de la solution, garantissant un code propre, structuré et modulaire pour le jury SupNum.*

*Le fichier Resultat.txt contient la resultat de test de 124 probleme qui on dans le dossier j60.sm, c'est resultat son obtient pour une limite de temps 108 Seconds.*
