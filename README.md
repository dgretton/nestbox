# Nestbox

Nestbox is a networked consensus coordinate system alignment tool. It provides a framework for aligning and optimizing multiple 3D coordinate systems across devices based on shared measurements and observations.

## Overview

Nestbox is designed to solve the problem of aligning multiple coordinate systems in a distributed environment. It's particularly useful in scenarios where different sensors or systems are observing the same features but from different perspectives or with different levels of uncertainty.

### Key Components

1. **Aligner**: The core optimization engine that aligns multiple coordinate systems based on shared observations.

2. **Coordinate Systems**: Representations of different frames of reference, each with its own origin and orientation.

3. **Measurements**: Data points observed by devices, including position and uncertainty information.

4. **Features**: Identifiable points or objects in space that can be observed across multiple coordinate systems.

5. **Daemon**: A background process that manages the alignment process and handles communication between different parts of the system.

6. **API**: Interfaces for interacting with the Nestbox system, allowing for the creation of coordinate systems, addition of measurements, and retrieval of alignment results.

7. **Visualizer**: A 3D visualization tool for displaying the state of the system, including the alignments of coordinate systems and the measurements of shared features.

8. **Simulator**: A lightweight toolkit for creating uncertain measurement data, including an *Environment* and different types of *Observers* like cameras or 3D trackers, useful for setting up known geometry and testing.


## Conceptual Workflow

1. Create multiple coordinate systems, each representing a different frame of reference.
2. Define features in the environment that can be observed in multiple coordinate systems.
3. Feed measurements of shared features into the system, including position and uncertainty information.
4. The aligner optimizes the relative positions and orientations of the coordinate systems to best fit all observations.
5. Visualize the results to see how the different coordinate systems align.
6. Use optimized numerical transformations between the aligned coordinate systems in your application.

## Project Structure

- `python/`: Contains the core Python implementation of Nestbox.
- `csharp/`: C# implementation for cross-platform support.
- `protos/`: Protocol buffer definitions for data structures.
- `desktop/`: Desktop application for Nestbox config, monitoring, and visualization.
- `visualizer/`: Browser-based visualization.
- `ui-core/`: Shared UI components and 3D visualization tools used by both web and desktop versions.
- `docs/`: Documentation and notes.

## Current State

Nestbox is currently in development. The system is not yet ready for production use. The optimization algorithms, API, and visualization capabilities are being refined and expanded.

## Contributing

As Nestbox is still in active development, we welcome contributions and feedback. Please feel free to open issues for bugs or feature requests. 
