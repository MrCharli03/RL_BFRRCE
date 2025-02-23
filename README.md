# Aprendizaje en entornos complejos

## Información
- **Alumnos:** Becerra Fernández, Esteban; Cruzado Esteban, Carlos; Ruzhytska Ruzhytska, Anastasiya
- **Asignatura:** Extensiones de Machine Learning
- **Curso:** 2024/2025
- **Grupo:** BFARCE

## Descripción
Esta práctica tiene como objetivo abordar el aprendizaje en entornos complejos, donde la toma de decisiones se desarrolla en contextos que no son estáticos. Se plantea la necesidad de enfrentarse a problemas en los que el modelo del entorno (transiciones, recompensas y política óptima) no es conocido, por lo que se recurre a la experiencia del agente para actualizar su política. Para ello, se trabajará con el entorno Gymnasium, permitiendo tanto el estudio de agentes que usan técnicas basadas en episodios (Monte Carlo, on-policy y off-policy) como algoritmos de diferencias temporales (SARSA, Q-Learning y variantes con aproximaciones como SARSA semi-gradiente y Deep Q-Learning).

## Estructura
El repositorio se organiza de la siguiente forma:
- **src/**: Código fuente (.py) de los agentes y algoritmos implementados.
- **docs/**: Documentación adicional e informes en formato PDF.
- **data/**: Datos utilizados en los experimentos (si procede).
- **README.md**: Este archivo.
- **main.ipynb**: Notebook principal que enlaza y organiza la ejecución de los diferentes experimentos.
- MonteCarloTodasLasVistas_Gymnasium.ipynb (cambiar a notebook 1) -> Aprendizaje en entornos complejos

## Instalación y Uso
Para poder observar los experimentos realizados así como el estudio llevado a cabo se recomienda abrir en Colab el documento "main.ipynb", ejecutarlo para importar los archivos necesarios, y desde el mismo navegar por los otros notebooks implementados, titulados "Notebook1.ipynb", "Notebook2.ipynb", "Notebook3.ipynb", "Notebook4.ipynb". (A cambiar tbn)

## Tecnologías Utilizadas
- **Lenguaje:** Python
- **Entorno de Ejecución:** Jupyter Notebooks / Google Colab
- **Librerías:** Gymnasium (https://gymnasium.farama.org/), NumPy, Matplotlib, tqdm.
