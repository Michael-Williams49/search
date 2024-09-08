# Search Algorithm Demonstration

This project provides both a Tkinter-based desktop application and a Flask-based web application to demonstrate various search algorithms on graphs. It's designed as an educational tool to help visualize how different search algorithms work.

## Introduction

The project consists of two main components:

1. A Tkinter GUI application (`search.py`) that allows users to interact with graphs and visualize search algorithms in real-time.
2. A Flask web application (`app.py`) that provides similar functionality through a web interface.

Both applications use a custom `Graph` class to represent and manipulate graph structures, and implement various search algorithms including Dijkstra's algorithm, Greedy search, and A* search.

## Features

- Interactive graph visualization
- Multiple pre-defined graph structures (Tree, Comparison, Admissibility, Extra)
- Random graph generation
- Step-by-step execution of search algorithms
- Support for different heuristics (Euclidean, Manhattan)
- Ability to change start and end nodes

## Installation

1. Clone this repository or download the source code
2. Unzip the archive and enter the project folder:

```sh
unzip search.zip
cd search
```

3. Create a virtual environment and activate it:

```sh
conda create -n search flask matplotlib networkx tkinter
conda activate search
```

## Usage

### Tkinter Application

To run the Tkinter application:

```sh
python search.py
```

Use the GUI to select different graphs, algorithms, and heuristics. Click "Next Step" to see the algorithm progress step-by-step.

### Web Application

To run the Flask web application:

```sh
flask run
```

Then open a web browser and navigate to `http://localhost:5000`. The web interface allows you to interact with the graphs and algorithms similarly to the Tkinter application.