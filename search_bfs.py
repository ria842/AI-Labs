import heapq


def manhattan_distance(a, b):
    """Calculate Manhattan distance between two positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def find_position(grid, symbol):
    """Find the position of S or G in the grid."""
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == symbol:
                return (r, c)
    return None


def reconstruct_path(parent, goal):
    """Reconstruct the path from S to G."""
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path


def astar(grid):
    start = find_position(grid, 'S')
    goal = find_position(grid, 'G')

    if start is None or goal is None:
        return None, 0

    # Priority queue:
    # (f(n), g(n), state)
    frontier = []

    # Cost from S to each state
    g_cost = {start: 0}

    # Parent of each state, used to reconstruct the path
    parent = {start: None}

    # States that have already been expanded
    expanded_states = set()

    # f(start) = g(start) + h(start)
    h = manhattan_distance(start, goal)
    heapq.heappush(frontier, (h, 0, start))

    while frontier:

        # Get state with smallest f(n)
        f, current_g, current = heapq.heappop(frontier)

        # Ignore an outdated entry
        if current_g != g_cost[current]:
            continue

        # Do not expand a state more than once
        if current in expanded_states:
            continue

        expanded_states.add(current)

        # Goal test
        if current == goal:
            path = reconstruct_path(parent, goal)
            return path, len(expanded_states)

        r, c = current

        # Up, Down, Left, Right
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1)
        ]

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc

            # Check boundaries
            if nr < 0 or nr >= len(grid):
                continue
            if nc < 0 or nc >= len(grid[0]):
                continue

            # Cannot move through obstacles
            if grid[nr][nc] == '#':
                continue

            neighbor = (nr, nc)

            # Every movement costs 1
            new_g = current_g + 1

            # If this is a new state or a cheaper path was found
            if neighbor not in g_cost or new_g < g_cost[neighbor]:

                g_cost[neighbor] = new_g
                parent[neighbor] = current

                # Manhattan heuristic
                h = manhattan_distance(neighbor, goal)

                # A* evaluation function
                f = new_g + h

                heapq.heappush(
                    frontier,
                    (f, new_g, neighbor)
                )

    # No path exists
    return None, len(expanded_states)


# --------------------------------------------------
# INPUT
# --------------------------------------------------

rows, cols = map(int, input("Enter rows and columns: ").split())

grid = []

print("Enter the grid:")
for _ in range(rows):
    grid.append(input().strip())


# --------------------------------------------------
# A* SEARCH
# --------------------------------------------------

path, expanded = astar(grid)


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

if path is None:
    print("No path exists.")
else:
    print("Path found:")
    print(path)

    # Number of movements
    print("Path length:", len(path) - 1)

print("States expanded:", expanded)

