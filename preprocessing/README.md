# Vorverarbeitung der Rohdaten

> Ich stelle per Mail alle nötigen Daten zur Verfügung, um jeden Schritt der Vorverarbeitung unabhängig ausführen zu können. Dafür muss die ZIP-Datei in den `data`-Ordner entpackt werden.

Um die Rohdaten zu verarbeiten, müssen diese im `data`-Ordner liegen. Danach werden durch die verschiedenen Skripte die Rohdaten vorverarbeitet und in verschiedenen Schritten zu Trainingsdaten umgewandelt. Die Namensgebung ist in der `env.py`-Datei zu finden und kann dort auch geändert werden.

Zunächst müssen die Rohdaten gefiltert werden. Mit dem `clean_and_filter.py`-Skript können diese gefiltert werden. Hier muss der Pfad noch manuell angegeben werden. Bitte folgenden Befehl ausführen:

```bash
    python clean_and_filter.py process ./data/raw_data.csv ./data/clean_and_filtered_data.csv
```

Zudem muss noch folgender Befehl ausgeführt werden, um eine Rides-Datei zu erstellen:

```bash
    python clean_and_filter.py rides ./data/clean_and_filtered_data.csv ./data/rides_data.csv
```

Sind beide Skripte ausgeführt, kann auch das erste Notebook ausgeführt werden: Zunächst `create_distance_matrix.ipynb`.

Die anderen Notebook-Dateien können auch im Nachhinein ausgeführt werden: `final_group_clustering.ipynb` sowie danach `create_train_validation_data.ipynb`.

Damit sollten die Daten fertig verarbeitet sein.
