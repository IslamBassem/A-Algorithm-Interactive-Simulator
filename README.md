# A* Pathfinding Visualizer

An interactive Python-based visualization of the A* (A-Star) pathfinding algorithm built using Matplotlib.

This project demonstrates how A* explores a grid to find the shortest path between two points while considering obstacles and weighted cells. It provides a real-time, visual understanding of heuristic search and path optimization.

## Features

- Fixed 20×20 grid (400 nodes)
- Interactive wall creation and removal
- Dynamic start and goal node placement
- Weighted cells with adjustable costs
- Real-time animation of node exploration
- Visualization of the final shortest path
- Adjustable simulation speed

## Controls

- **Left Click:** Add wall  
- **Right Click:** Remove wall  
- **S + Click:** Set start node  
- **G + Click:** Set goal node  
- **W + Click:** Apply weight (based on slider)  

## Technologies Used

- Python
- Matplotlib
- NumPy

## Overview

The implementation uses the A* algorithm with a Manhattan distance heuristic and a priority queue (min-heap) to efficiently compute the shortest path. The visualization highlights both the exploration process and the final optimal route.

## Purpose

This project is designed for educational purposes and helps in understanding:
- Pathfinding algorithms
- Heuristic-based search
- Grid-based problem solving

## Author
**Eng/ Islam Bassem**
