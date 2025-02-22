# Aprendizaje en entornos complejos

## Información
- **Alumnos:** Becerra Esteban, Cruzado Carlos, Ruzhytska Anastasia
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** BFARCE

## Descripción
Esta práctica tiene como objetivo abordar el aprendizaje en entornos complejos, donde la toma de decisiones se desarrolla en contextos que no son estáticos. Se plantea la necesidad de enfrentarse a problemas en los que el modelo del entorno (transiciones, recompensas y política óptima) no es conocido, por lo que se recurre a la experiencia del agente para actualizar su política. Para ello, se trabajará con el entorno Gymnasium, permitiendo tanto el estudio de agentes que usan técnicas basadas en episodios (Monte Carlo, on-policy y off-policy) como algoritmos de diferencias temporales (SARSA, Q-Learning y variantes con aproximaciones como SARSA semi-gradiente y Deep Q-Learning).

## Estructura
El repositorio se organiza de la siguiente forma:
- **src/**: Código fuente (.py) de los agentes y algoritmos implementados.
- **notebooks/**: Notebooks (.ipynb) con la implementación y experimentos en el entorno Gymnasium, incluyendo:
  - Notebook para la familiarización y experimentación básica con Gymnasium.
  - Notebook(s) para el estudio comparativo de los algoritmos (Monte Carlo, SARSA, Q-Learning, etc.).
- **docs/**: Documentación adicional e informes en formato PDF.
- **data/**: Datos utilizados en los experimentos (si procede).
- **README.md**: Este archivo.
- **main.ipynb**: Notebook principal que enlaza y organiza la ejecución de los diferentes experimentos.

## Instalación y Uso
1. Clonar el repositorio en su entorno local o directamente en Google Colab.
2. Abrir el archivo **main.ipynb** en Google Colab (o en el entorno de Jupyter Notebook).
3. Ejecutar las celdas iniciales que instalan las dependencias y copian el repositorio.
4. Navegar entre los notebooks para revisar y ejecutar los experimentos.  
   
*Nota:* Asegúrese de que todas las celdas se ejecuten sin errores para garantizar la reproducibilidad de los resultados.

## Tecnologías Utilizadas
- **Lenguaje:** Python
- **Entorno de Ejecución:** Jupyter Notebooks / Google Colab
- **Librerías:** Gymnasium (https://gymnasium.farama.org/), NumPy, Matplotlib, entre otras utilizadas para el desarrollo de los agentes de aprendizaje por refuerzo.
