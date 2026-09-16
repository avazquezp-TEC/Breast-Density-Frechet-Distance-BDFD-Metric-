# Breast-Density-Frechet-Distance-BDFD-Metric
````bash
git clone https://github.com/avazquezp-TEC/Breast-Density-Frechet-Distance-BDFD-Metric-.git
````


Para ejecutar cualquiera de los archivos desarrollados en esta pagina hay que invocar al entorno de desarrollo: 
```bash
source ~/mac_metal311/bin/activate
```
Para linux usar entorno de desarrollo diferente `python3.11 -m venv ~/torch_venv` a tensorflow ya que desconfigura el programa:
````bash
source ~/torch_venv/bin/activate
````
y cambiar la ruta de csv
```python
ruta_csv = Path("../Experimentos/Mammo_Bench_v2/mammo-bench.csv")
```


## A. Prueba de inferencia convext tiny 
1. Se desarrolló un programa de inferencia del convnext original `a_prueba_convnext.py`. version tiny

```bash
python a_prueba_convnext.py --image ../../../Sergio/SRIR/DATASET/DIV2K_train_HR/0189.png
```
## B. Entrenamiento tiny

2. Se esta desarrollando el train quitando el header de 1000 categorias por uno de 4 correspondiente al densidad mamaria y manteniendo descongelado el backbone para hacer ajuste fino.
El programa es `b_train_convnext.py`
Para usarlo: 
```bash 
python b_train_convnext.py
```
Este fue el resultado al aplicar:
1. WeightedRandomSampler ya que mis clases estan desbalanceadas.
2. Tengo ReduceLROnPlateau para ajustar el LR
3. Transfer learning usando los ConvNeXt_Tiny_Weights.DEFAULT
4. Pre-procesamiento de imagen de imagefNet.
5. Early stopping con 7 de paciencia (falta ajustar criterio)

```bash
Validation Stage (mejor modelo) -> Loss: 0.8946 | Accuracy: 60.26% | Precision(macro): 59.65% | Recall(macro): 62.75% | F1(macro): 60.42%
```

Vamos a agregar:
self-supervised te permite aprovechar las 2 000 imagenes sin etiqueta y las 7 000 MLO para preentrenar, y luego afinar en CC.

3. Se desarrollará inferencia para prueba. usando mismo programa base. cambiando args para indicar que modelo usar
4. Se desarrollará programa que acceda a backbone. reemplazando header por iden con el fin de acceder al vector de caracteristicas del backbone y poder usar FRECHET

## C. Entrenamiento tiny
Se agrego self supervised a los archivos :`c_dataset_manage.py` `c_train_convnext.py` `c_train_mae_convnextv2.py`. 

`c_dataset_manage.py` y `c_train_convnext.py` son modificaciones de las versiones anteriores modificados unicamente para adoptar SSL. 
`c_train_mae_convnextv2.py` por otro lado, es un archivo completamente nuevo que emplea las 12,521 imagenes restantes del data set y emplea SSL por enmascaramiento para aprender sobre convNext V2. Posteriormente se haran pruebas haciendo finetunning con el proceso anterior para ver si mejoran los resultados
Este archivo cuenta con argsparse:

--csv ruta del archivo

--epochs

--patience numero de elementos para paciencia

--batch-size

--lr

--mas-ratio

--out directorio donde se guarda el archivo


Hasta el momento lleva 39 epocas 6horas

Empezo a las 10:52 am Termino 0:15 am = 13h: 23min

ahora se ejecuta c_train_convnext.py para hacer finetunning del clasificador.
Ejecuto 13 epocas.

precision = TruePositives/(True Positives+FalsePositives)
FalsePositives= No es x, predijo x

de todas las veces que el modelo dijo que es x, cuantas realmente eran x.

recall = TP/(TP+FN)

False negative: si pertenecen a x, pero NO los clasifico como x.

True → el modelo acerto. 
False → El modelo no acerto.
Positive → el modelo predijo que sí era X.
Negative → el modelo predijo que no era X.

```bash
Validation Stage (mejor modelo) -> Loss: 0.9627 | Accuracy: 56.66% | Precision(macro): 58.91% | Recall(macro): 60.17% | F1(macro): 56.83%
```

ahora con SSL:
```bash
Validation Stage (mejor modelo) -> Loss ↓: 0.8177 (se equivoco menos)
Accuracy ↑: 64.6486% (total de predicciones correctas sobre total de ejemplos)
Precision(macro)↑: 64.6074% (predicciones positivas que realmentente eran correctas)
Recall(macro)↑: 67.4706%  (predicciones correctas sobre positivas unicamente)
 F1(macro)↑: 65.1843% (armónica entre precisión y recall)
2026-09-14 12:30:27,504 - INFO:__main__: Matriz de confusión (filas=real, columnas=predicho, orden A,B,C,D):
[[151  59  10   8]
 [ 77 325 167  18]
 [  5  35 280  82]
 [  4   6  37 173]]
 ```
 le ejecución ssl se realizó con
 batch 32,
 lr = 1e-4 con CosineAnnealingLR
 mask ratio 0.7
 max epochs 200 patience 14 delta 1e-4
 se entreno por 105 epocas y se restaura 105-14
 seed 42

 finetunning
seed: int = 42,
    max_epochs: int = 100,
    patience: int = 10,min_delta = 1e-4
    batch_size: int = 16
    backbone_lr=1e-5, head_lr=1e-4, ReduceLROnPlateau

 Procedemos a usar este sistema para hacer pruebas con frechet o proyección MDS O MMD con kernel distinto

 ## evaluación de frechet

 
 python d_frechet_metrics.py \
    --original-path /ruta/a/imagenes_reales \
    --artificial-path /ruta/a/imagenes_generadas \
    --checkpoint convnextv2_tiny_custom.pth \
    --out-dir resultados_frechet/