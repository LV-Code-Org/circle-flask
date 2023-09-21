from __future__ import annotations


class Line:
    def __init__(self, start: tuple, end: tuple):
        try:
            self.slope = (end[1] - start[1]) / (end[0] - start[0])
        except ZeroDivisionError:
            self.slope = 0.001
        self.bias = start[1] - (self.slope * start[0])
        self.start = start
        self.intersections = []
        self.end = end


class Game:
    def __init__(self, allLines: list[list], currentIntersections: list[list]):
        self.allLines = allLines
        self.allLinesNodes = []
        self.solution = []
        self.found = False
        self.recursionCounter = 0
        self.sc = 0
        self.currentIntersections = currentIntersections

    def update(self) -> None:
        """Updates nodes and intersections."""
        self.update_intersections()
        self.allLinesNodes = [[line.start, *line.intersections, line.end] for line in self.allLines]

    def run(self) -> None:
        """Runs the game."""
        self.update()
        self.compute_click()
        return {
            "pentagon_found": self.found,
            "solution": self.solution,
            "intersections": self.get_intersections(),
            "score": self.sc
        }
    
    def get_intersections(self) -> list:
        allIntersections = []
        for line in self.allLines:
            allIntersections.append(line.intersections)
        flattened_list = list(set([item for sublist in allIntersections for item in sublist]))
        return flattened_list

    def is_same_node(self, node: tuple, all_nodes: list[tuple]) -> bool:
        """Checks if a node is the same node as one that is already on the board."""
        from math import sqrt
        x2, y2 = node
        for n in all_nodes:
            x1, y1 = n
            # Check if node is outside of the white circle on screen
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= 4:
                self.sc -= 1
                return [True, n]
        return [False, None]

    def update_intersections(self) -> None:
        import itertools
        """Updates the intersections property of each Line in allLines."""
        intersections = []
        for i, line1 in enumerate(self.allLines):
            for line2 in self.allLines[i + 1:]:
                # Find all intersections
                inter = find_intersection(line1, line2)
                isn = self.is_same_node(inter, intersections)
                flattened_endpoints = list(itertools.chain(*[[l.start, l.end] for l in self.allLines]))

                # if not self.is_same_node(inter, flattened_endpoints):
                if isn[0]:
                    x, y = isn[1]
                    line1.intersections.append((x, y))
                    line2.intersections.append((x, y))
                    # TODO: Implement score logic
                    print("ADD SCORE")
                    self.sc += 1 # score increase
                else:
                    if inter is not None:
                        x, y = inter
                        # Check if the intersection is valid (within the constraints of the circle)
                        distance_to_center = ((x - 250) ** 2 + (y - 250) ** 2) ** 0.5
                        if distance_to_center <= 200:
                            if inter not in self.currentIntersections:
                                print("ADD SCORE")
                                self.sc += 1 # score increase
                            line1.intersections.append((x, y))
                            line2.intersections.append((x, y))
                            intersections.append(inter)

        # Sort each line's intersections by x-value
        intersections.sort(key=lambda point: point[0])
        for line in self.allLines:
            rev = True if line.end[0] < line.start[0] else False
            line.intersections.sort(key=lambda point: point[0], reverse=rev)

    def find_adjacent_nodes(self, node: tuple) -> list[tuple]:
        """Finds all adjacent nodes for a given node."""
        from itertools import chain
        final = []
        for g in self.allLinesNodes:
            try:
                idx = g.index(node)
                if g[idx] == g[0]:
                    final.append([(g[idx + 1][0], g[idx + 1][1])])
                elif g[idx] == g[-1]:
                    final.append([(g[idx - 1][0], g[idx - 1][1])])
                else:
                    final.append([g[idx - 1], g[idx + 1]])
            except ValueError:
                ...
        return list(chain.from_iterable(final))

    def node_dictionary(self) -> dict[tuple, list[tuple]]:
        """Returns a dictionary where each node is a key and its adjacent nodes are the value."""
        unique_nodes = list(set([item for sublist in self.allLinesNodes for item in sublist]))
        node_guide = {}
        for n in unique_nodes:
            node_guide[n] = self.find_adjacent_nodes(n)
        return node_guide

    def get_immediate_neighbors(self, node: tuple) -> list[tuple]:
        """Returns a list of all immediate neighbor nodes."""
        return self.node_dictionary()[node]

    def colinearity_check(self, shape: list) -> bool:
        """Check if a line exists that contains three of the points in the shape."""
        for line in self.allLinesNodes:
            counter = 0
            for point in shape:
                if point in line:
                    counter += 1
            if counter > 2:
                return True
        return False

    def check_pentagon(self, origin: tuple, parent: tuple, depth: int, shape: list, myself: tuple | None) -> list[bool, list]:
        """Recursive function that checks if a pentagon exists. Updates the *solution* with a solution if found."""
        # Exit conditions
        if self.found:
            return [True, shape]
        if depth > 5:
            return [False, shape]
        if myself == origin:
            if depth == 5:
                self.found = True
                self.solution = shape.copy()
                return [True, shape]
            else:
                return [False, shape]
        if myself is None:
            return [False, shape]
        if myself == parent:
            return [False, shape]
        if self.colinearity_check(shape):
            return [False, shape]

        # Recursive loop
        immediate_neighbors = self.get_immediate_neighbors(myself)
        for neighbor in immediate_neighbors:

            my_shape = shape.copy()
            if neighbor not in my_shape or (neighbor == origin and depth > 1):
                my_shape.append(neighbor)
                self.recursionCounter += 1
                self.check_pentagon(origin=origin, parent=myself, depth=depth + 1, myself=neighbor, shape=my_shape)

            # Leaf node detection
            if len(immediate_neighbors) == 1 and parent == neighbor:
                neighbor = None
                self.recursionCounter += 1
                self.check_pentagon(origin=origin, parent=myself, depth=depth + 1, myself=neighbor, shape=my_shape)

    def compute_click(self) -> None:
        """Checks if there is a pentagon."""

        self.recursionCounter = 0  # Reset recursion counter

        # Get all nodes
        all_nodes = list(self.node_dictionary().keys())
        found = 0

        # Check pentagon
        for current_node in all_nodes:
            shape = [current_node]
            if found:
                break

            immediate_neighbors = self.get_immediate_neighbors(current_node)
            for neighbor in immediate_neighbors:
                my_shape = shape.copy()
                my_shape.append(neighbor)
                depth = 0
                origin = myself = current_node
                x = self.check_pentagon(origin=origin, parent=myself, depth=depth + 1, myself=neighbor, shape=my_shape)
                if x:
                    if x[0]:
                        found = 1


def find_intersection(line_1: Line, line_2: Line) -> list | None:
    """Finds the intersection between two lines."""
    import numpy as np
    if line_1.slope == line_2.slope:
        return None
    a = np.array([[line_1.slope, -1], [line_2.slope, -1]])
    b = np.array([-line_1.bias, -line_2.bias])
    return list(np.linalg.solve(a, b))

def run_game(allLines: list[list], currentIntersections: list[list]) -> dict:
    game = Game([Line(tuple(x[0]), tuple(x[1])) for x in allLines], currentIntersections)
    result = game.run()
    return result

# run_game([
#     [(134, 86), (99, 381)],
#     [(52, 220), (340, 428)],
#     [(223, 448), (405, 123)],
#     [(447, 283), (229, 51)],
#     [(366, 87), (79, 145)],
# ])

