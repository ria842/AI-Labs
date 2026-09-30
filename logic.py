"""
Simple planning agent.

State representation:
    A state is a frozenset of propositions (strings), e.g.
        frozenset({"At(Robot,A)", "At(Package,A)"})

Action representation:
    Each action has:
        - name
        - pos_preconditions: propositions that must be TRUE in the state
        - neg_preconditions: propositions that must be FALSE (absent) in the state
        - pos_effects: propositions to ADD to the state after applying the action
        - neg_effects: propositions to REMOVE from the state after applying the action

An action is applicable in state S if:
    pos_preconditions subset of S   AND   neg_preconditions disjoint from S

Applying an action:
    S' = (S - neg_effects) | pos_effects

Search:
    Breadth-first search over the state graph, from the initial state,
    until a state satisfying the goal (goal subset of state) is reached.
    BFS guarantees the SHORTEST plan (fewest actions) is found first.
"""

from collections import deque


class Action:
    def __init__(self, name, pos_preconditions=None, neg_preconditions=None,
                 pos_effects=None, neg_effects=None):
        self.name = name
        self.pos_preconditions = frozenset(pos_preconditions or [])
        self.neg_preconditions = frozenset(neg_preconditions or [])
        self.pos_effects = frozenset(pos_effects or [])
        self.neg_effects = frozenset(neg_effects or [])

    def is_applicable(self, state):
        # Logical component: S |= Preconditions(a)
        return self.pos_preconditions <= state and self.neg_preconditions.isdisjoint(state)

    def apply(self, state):
        # S' = Apply(S, a)
        return (state - self.neg_effects) | self.pos_effects

    def __repr__(self):
        return self.name


def goal_satisfied(state, goal):
    return goal <= state


def plan(initial_state, actions, goal, verbose=True):
    """Breadth-first search for a sequence of applicable actions
    that transforms initial_state into a state satisfying goal.
    Returns (plan_actions, states_visited) or (None, None) if no plan exists.
    """
    initial_state = frozenset(initial_state)
    goal = frozenset(goal)

    if goal_satisfied(initial_state, goal):
        return [], [initial_state]

    frontier = deque()
    frontier.append((initial_state, [], [initial_state]))
    visited = {initial_state}

    while frontier:
        state, path, state_path = frontier.popleft()

        for action in actions:
            if not action.is_applicable(state):
                continue  # Logic: precondition check fails -> action not tried
            new_state = action.apply(state)  # Generate successor state
            if new_state in visited:
                continue
            new_path = path + [action]
            new_state_path = state_path + [new_state]

            if goal_satisfied(new_state, goal):
                if verbose:
                    print("Plan found!")
                return new_path, new_state_path

            visited.add(new_state)
            frontier.append((new_state, new_path, new_state_path))

    if verbose:
        print("No plan found")
    return None, None


def print_plan(actions_used, states, initial_state, goal):
    print("Initial state:", sorted(initial_state))
    print("Goal:", sorted(goal))
    if actions_used is None:
        print("Result: No plan found")
        return
    print(f"Result: Plan found with {len(actions_used)} action(s)")
    print(f"  S0: {sorted(states[0])}")
    for i, (a, s) in enumerate(zip(actions_used, states[1:]), start=1):
        print(f"  a{i} = {a.name}")
        print(f"  S{i}: {sorted(s)}")


def make_warehouse_actions(include_pickup=True):
    """Builds the action set for the warehouse problem.
    include_pickup=False is used for Test B (impossible problem)."""
    actions = []

    # Move actions between connected locations
    connections = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]
    for x, y in connections:
        actions.append(Action(
            name=f"Move(Robot,{x}->{y})",
            pos_preconditions=[f"At(Robot,{x})"],
            pos_effects=[f"At(Robot,{y})"],
            neg_effects=[f"At(Robot,{x})"],
        ))

    if include_pickup:
        for loc in ["A", "B", "C"]:
            actions.append(Action(
                name=f"PickUp(Package,{loc})",
                pos_preconditions=[f"At(Robot,{loc})", f"At(Package,{loc})"],
                pos_effects=["Holding(Package)"],
                neg_effects=[f"At(Package,{loc})"],
            ))

    for loc in ["A", "B", "C"]:
        actions.append(Action(
            name=f"Drop(Package,{loc})",
            pos_preconditions=[f"At(Robot,{loc})", "Holding(Package)"],
            pos_effects=[f"At(Package,{loc})"],
            neg_effects=["Holding(Package)"],
        ))

    return actions


if __name__ == "__main__":
    initial_state = {"At(Robot,A)", "At(Package,A)"}
    goal = {"At(Package,C)"}

    print("=" * 60)
    print("TEST A: Solvable problem (original warehouse)")
    print("=" * 60)
    actions = make_warehouse_actions(include_pickup=True)
    actions_used, states = plan(initial_state, actions, goal)
    print_plan(actions_used, states, initial_state, goal)

    print()
    print("=" * 60)
    print("TEST B: Impossible problem (PickUp action removed)")
    print("=" * 60)
    actions_no_pickup = make_warehouse_actions(include_pickup=False)
    actions_used_b, states_b = plan(initial_state, actions_no_pickup, goal)
    print_plan(actions_used_b, states_b, initial_state, goal)

    print()
    print("=" * 60)
    print("TEST C: Irrelevant action (robot can reach C, package stays at A)")
    print("=" * 60)
    # Same as Test A's action set already includes Move(B->C), which lets the
    # ROBOT reach C without the package. The goal only cares about the
    # package's location, so this checks that the planner doesn't confuse
    # "robot at C" with "package at C".
    initial_state_c = {"At(Robot,A)", "At(Package,A)"}
    actions_used_c, states_c = plan(initial_state_c, actions, goal)
    print_plan(actions_used_c, states_c, initial_state_c, goal)
    # Sanity check: does the shortest plan actually carry the package, not
    # just move the robot to C?
    if actions_used_c:
        assert any("PickUp" in a.name for a in actions_used_c), \
            "Planner found a plan that never picks up the package!"
        assert any("Drop" in a.name for a in actions_used_c), \
            "Planner found a plan that never drops the package at the goal!"
        print("  Check passed: plan actually moves the PACKAGE, not just the robot.")   