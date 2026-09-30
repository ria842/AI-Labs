from collections import deque

# ------------------------------------------------------------
# Goal-Based Agent for Warehouse Navigation
# ------------------------------------------------------------
# Symbols used in the grid:
#   S -> Start position
#   G -> Goal position
#   # -> Obstacle / blocked cell
#   . -> Free cell
#   * -> Path found by the agent
#
# The agent uses Breadth-First Search (BFS) to find a
# collision-free path from S to G.
# ------------------------------------------------------------


def find_position(grid, symbol):
    """
    Find the position of a given symbol in the grid.

    Parameters:
        grid   : 2D list representing the warehouse
        symbol : symbol to search for ('S' or 'G')

    Returns:
        (row, column) of the symbol.

    Raises:
        ValueError if the symbol is not present.
    """
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == symbol:
                return row, col

    raise ValueError(f"{symbol} not found in the warehouse.")


def bfs(grid, start, goal):
    """
    Perform Breadth-First Search to find a shortest path
    from start to goal.

    Parameters:
        grid  : 2D warehouse grid
        start : (row, column) of starting position
        goal  : (row, column) of goal position

    Returns:
        A list containing the cells in the path from start
        to goal, or None if no path exists.
    """

    rows = len(grid)
    cols = len(grid[0])

    # Possible movements:
    # Up, Down, Left, Right
    directions = [
        (-1, 0),   # Up
        (1, 0),    # Down
        (0, -1),   # Left
        (0, 1)     # Right
    ]

    # Queue used by BFS.
    # Each entry contains the current cell.
    queue = deque([start])

    # Keep track of cells that have already been visited.
    visited = {start}

    # parent[cell] stores the cell from which 'cell' was reached.
    # This allows us to reconstruct the final path.
    parent = {start: None}

    while queue:

        # Remove the next cell from the front of the queue.
        current = queue.popleft()

        # Goal reached.
        if current == goal:
            return reconstruct_path(parent, goal)

        current_row, current_col = current

        # Explore all four neighboring cells.
        for dr, dc in directions:

            new_row = current_row + dr
            new_col = current_col + dc

            # Check whether the new position is inside the grid.
            if not (0 <= new_row < rows and 0 <= new_col < cols):
                continue

            # Ignore obstacles.
            if grid[new_row][new_col] == '#':
                continue

            neighbor = (new_row, new_col)

            # Ignore cells that have already been visited.
            if neighbor in visited:
                continue

            # Mark the cell as visited.
            visited.add(neighbor)

            # Store where we came from.
            parent[neighbor] = current

            # Add the cell to the BFS queue.
            queue.append(neighbor)

    # Queue became empty without reaching the goal.
    return None


def reconstruct_path(parent, goal):
    """
    Reconstruct the path from S to G using the parent dictionary.

    Parameters:
        parent : dictionary containing each cell's predecessor
        goal   : goal position

    Returns:
        List of positions from start to goal.
    """

    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    # The path was constructed backwards, so reverse it.
    path.reverse()

    return path


def display_path(grid, path):
    """
    Display the warehouse grid with the discovered path.

    Parameters:
        grid : original warehouse grid
        path : list of positions in the path
    """

    # Make a copy so that the original grid is not modified.
    result = [row[:] for row in grid]

    # Mark the path using '*'.
    # Keep S and G unchanged.
    for row, col in path:
        if result[row][col] not in ('S', 'G'):
            result[row][col] = '*'

    print("\nWarehouse with path:")
    for row in result:
        print(" ".join(row))


def main():
    """
    Main function implementing the goal-based warehouse agent.
    """

    # --------------------------------------------------------
    # Warehouse representation
    # --------------------------------------------------------
    #
    # S = Starting position
    # G = Goal position
    # # = Obstacle
    # . = Free space
    #
    warehouse = [
        ['S', '.', '.', '#', '.', '.', '.'],
        ['#', '#', '.', '#', '.', '#', '.'],
        ['.', '.', '.', '.', '.', '#', '.'],
        ['.', '#', '#', '#', '.', '#', '.'],
        ['.', '.', '.', '.', '.', '.', '.'],
        ['#', '#', '#', '#', '#', '#', '.'],
        ['.', '.', '.', '.', '.', '.', 'G']
    ]

    # Find the starting and goal positions.
    start = find_position(warehouse, 'S')
    goal = find_position(warehouse, 'G')

    print("Start:", start)
    print("Goal :", goal)

    # --------------------------------------------------------
    # Goal-based search
    # --------------------------------------------------------
    path = bfs(warehouse, start, goal)

    if path is None:
        print("\nNo collision-free path exists from S to G.")
    else:
        print("\nPath found!")

        # Print the sequence of positions.
        print("Path:")
        for position in path:
            print(position, end=" -> ")

        print("END")

        print("\nNumber of moves:", len(path) - 1)

        # Display the warehouse with the path marked.
        display_path(warehouse, path)


# ------------------------------------------------------------
# Program entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    main()