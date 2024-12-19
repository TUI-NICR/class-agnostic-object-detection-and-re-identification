# !!DEPRECATED!!

## Setup für Pipeline
Die benutzte Conda environment liegt unter /???/environment.yml. Zusätzlich muss noch MMDetection installiert werden. Außerdem werden noch zwei Packages benötigt, die für die Detektion und die Wiedererkennung verantwortlich sind. Diese können unter /???/packages gefunden werden. Congig Dateien und Checkpoints können ebenfalls unter /???/ gefunden werden.

## Pipeline
Der Code für die Pipeline befindet sich in der Datei "pipeline_reverse.py". Die Pipeline teilt ein Bild zunächst in Bildausschnitte ein und führt anschließend eine Detektion und eine nachfolgende Wiedererkennung auf den prädizierten Bounding Boxen durch. Optional kann vor der Detektion eine weitere Wiedererkennung auf den Bildausschnitten durchgeführt werden, um Bereiche des Bildes die keine Objekte enthalten früh auszuschließen und in der weiteren Verarbeitung Rechenzeit zu sparen. 

## Beispielaufruf ohne erste Wiedererkennung
```bash
python pipeline_reverse.py \
    --img-filepath /path/to/attach_benchmark/imgs_full/table/ \ 
    --confidence-threshold 0.075 \
    --output-path /path/to/results/ \
    --crop-height 650 \
    --crop-width 650 \
    --comparison-images /path/to/attach_benchmark/reid/comparison_all/ \
    --config-file /path/to/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
    --checkpoint-file /path/to/checkpoints/dino_epoch_20.pth \
    --boxes-reid2 5 \
    --eval \ (optinal)
    --visualize \ (optional)
    --crop-image-table (optinal)
```
Erklärung der Parameter:
- img-filepath: Pfad zu den Inputbildern
- confidence-threshold: nur pärdizierte Boxen des Detektors, deren Konfidenz größer ist werden beibehalten
- output-path: Pfad, wo die Ergebnisse gespeichert werden sollen
- crop-height, crop-width: Größe der Bildausschnitte die am Anfang erstellt werden
- comparison-images: Pfad zu den Vergleichsbildern für die Wiedererkennung
- config-file: Pfad zur Config des mmdet Models, welches benutzt wird
- checkpoint-file: Trained Model
- boxes-reid2: maximale Anzahl an Boxen die nach 2. Wiedererkennung jedem Vergleichsobjekt zugeordnet werden
- eval: Evaluierung der Ergebnisse falls Ground Truth Daten vorhanden sind
- visualize: Visualisierung der Ergebnisse
- crop-image-table: Bilder sind sehr groß + Randbereiche enthalten viele Dinge, wo unnötige Boxen erstellt werden, die auch noch in der Wiedererkennung "gematcht" werden --> nicht relevante Bereiche direkt ausschließen


## Beispielaufruf mit zusätzlicher Wiedererkennung
```bash
python pipeline_reverse.py \
    --reverse \
    --threshold-reid1 \
    --max-crops-reid1 \
    --img-filepath /path/to/attach_benchmark/imgs_full/table/ \ 
    --confidence-threshold 0.075 \
    --output-path /path/to/results/ \
    --comparison-images /path/to/attach_benchmark/reid/comparison_all/ \
    --config-file /path/to/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py \
    --checkpoint-file /path/to/checkpoints/dino_epoch_20.pth \
    --boxes-reid2 5 \
    --downscale-factor 1\
    --box-size 512\
    --eval \ (optinal)
    --visualize \ (optional)
    --crop-image-table (optinal)
```

Zusätzliche Parameter:
- reverse: leitet erste Wiedererkennung vor Detektion ein
- threshold-reid1: Nur Bildausschnitte die geringeren Wert (Vergleich mit den Vergleichsbildern) haben kommen für Detektion in Frage
- max-crops-reid1: maximale Anzahl an Bildausschnitten die pro Vergleichsbild ausgewählt werden für die Detektion
- downscale-factor: factor for rescaling the image for first reidentification
- box-size: size of boxes for detection 

## Beispielablauf der Pipeline
1. Die geladenen Bilder werden zunächst in Bildausschnitte unterteilt mit einem Sliding Window, wobei das Fenster sich immer um die Hälfte seiner Breite/Höhe verschiebt. 
2. (optional) Das Bild wird in voller oder reduzierter Auflösung in das Wiedererkennungsmodul gegeben. Dort wird die Cosinus-Distanz zwischen dem Bild und den Vergleichsobjekten berechnet, wobei im Wiedererkennungsnetzwerk das Global Average Pooling am Ende weggelassen wird, sodass die Ausgabe mehrere Feature Vektoren sind. Für jeden der Vektoren wird die Cosinus Distanz berechnet. Es kommen nur die Vektoren in Frage, die einen Wert geringer als "threshold-reid1" zu einem Vergleichsobjekt haben und es werden maximal "max-crops-reid1" Bildausschnitte pro Vergleichsobjekt ausgewählt. Zu den Vektoren werden im Postprocessing die entsprechenden Positionen im Originalbild berechnet. Für jede Position wird im Anschluss ein Bildasuschnitt für die nachfolgende Detektion berechnet. Dabei müssen noch nah beieinander liegende Feature Vektoren herausgefiltert werden (oftmals sind 3 oder mehr direkt nebeneinander liegende Vektoren unter dem Schwellwert - diese würden alle nahazu den gleichen Bildausschnitt für die Detektion erzeugen)
3. Die ausgewählten Bildausschnitte oder alle in 1. erstellten Ausschnitte, falls 2. nicht durchgeführt wurde werden in das Detektionsmodul gegeben. Dort wird eine normale Objektdetektion durchgeführt und alle Boxen mit Scores kleiner als der festgelegte "confidence-threshold" werden herausgefiltert.
4. Es folgt die Wiedererkennung auf den prädizierten Boxen, wieder mit den Bildern der Vergleichsobjekte. Anschließend erfolgt eine erneute Filterung wie in 1., nur mit anderen Parametern für Schwellwert und maximale Anzahl an Boxen. Übrig bleiben die Bounding Boxen die die höchste Ähnlichkeit zu einem der Vergleichsobjekte besitzen.
5. (optional) Visualisierung der Bildausschnitte die in 2. ausgewählt werden ist möglich und die Visualisierung der finalen Boxen nach der Wiedererkennung. Außerdem können auch nur die TP Boxen nach Detektion und Wiedererkennung visualisert werden, dies geht allerdings nur wenn Ground Truths vorhanden sind.

## Anmerkungen
- Sowohl DINO als auch RTMDet wurden vorher auf dem LVIS Datensatz trainiert.
- RTMDet liefert etwas schlechtere Boxen (hat niedriegeren Recall) und dazu sind es auch noch deutlich mehr Boxen, weshalb die Wiedererkennung länger dauert.


## Pipeline mit Elephant Of Object Detection
- benötigt Detectron2
- funktioniert theroretisch auch, ist aber weniger weit entwickelt, da mit keiner Konfiguration ein ansatzweise guter Recall erreicht werden konnte 
- zudem ist es super langsam 

## Training
- Trainingskonfigurationen liegen im Verzeichnis /trainings_configs. Die Pfade zu den Bildern und Annotation-Files müssten noch entsprechend angepasst werden.