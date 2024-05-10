vertices = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]
edges = [
            ((0, 0, 0), (1, 0, 0)),  # Edge along the x-axis on the bottom face
            ((0, 0, 0), (0, 1, 0)),  # Edge along the y-axis on the bottom face
            ((0, 0, 0), (0, 0, 1)),  # Edge along the z-axis from the origin
            ((1, 0, 0), (1, 1, 0)),  # Parallel to y-axis at x=1 on the bottom face
            ((1, 0, 0), (1, 0, 1)),  # Parallel to z-axis at x=1, y=0
            ((0, 1, 0), (1, 1, 0)),  # Parallel to x-axis at y=1 on the bottom face
            ((0, 1, 0), (0, 1, 1)),  # Parallel to z-axis at x=0, y=1
            ((0, 0, 1), (1, 0, 1)),  # Parallel to x-axis at z=1 on the top face
            ((0, 0, 1), (0, 1, 1)),  # Parallel to y-axis at z=1 on the top face
            ((1, 1, 0), (1, 1, 1)),  # Parallel to z-axis at x=1, y=1
            ((1, 0, 1), (1, 1, 1)),  # Parallel to y-axis at x=1 on the top face
            ((0, 1, 1), (1, 1, 1))   # Parallel to x-axis at y=1, z=1 on the top face
        ]