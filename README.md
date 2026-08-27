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



## 1. Prueba de inferencia convext tiny 
1. Se desarrolló un programa de inferencia del convnext original `a_prueba_convnext.py`. version tiny

```bash
python a_prueba_convnext.py --image ../../../Sergio/SRIR/DATASET/DIV2K_train_HR/0189.png
```
## 2. Entrenamiento tiny

2. Se esta desarrollando el train quitando el header de 1000 categorias por uno de 4 correspondiente al densidad mamaria y manteniendo descongelado el backbone para hacer ajuste fino.
El programa es `b_train_convnext.py`
Para usarlo: 
```bash 
python b_train_convnext.py
````



3. Se desarrollará inferencia para prueba. usando mismo programa base. cambiando args para indicar que modelo usar
4. Se desarrollará programa que acceda a backbone. reemplazando header por iden con el fin de acceder al vector de caracteristicas del backbone y poder usar FRECHET
