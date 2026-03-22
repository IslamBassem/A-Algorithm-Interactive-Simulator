import heapq
import itertools
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import matplotlib.colors as mcolors


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_walkable_neighbors(col, row, grid):
    rows, cols = len(grid), len(grid[0])
    for dc, dr in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        nc, nr = col + dc, row + dr
        if 0 <= nc < cols and 0 <= nr < rows and grid[nr][nc] != -1:
            yield nc, nr, grid[nr][nc]


def run_astar(grid, source, target):
    rows, cols = len(grid), len(grid[0])
    if not (0 <= source[0] < cols and 0 <= source[1] < rows) or grid[source[1]][source[0]] == -1:
        return [], float('inf'), []
    if not (0 <= target[0] < cols and 0 <= target[1] < rows) or grid[target[1]][target[0]] == -1:
        return [], float('inf'), []

    counter = itertools.count()
    open_set = [(manhattan_distance(source, target), next(counter), source)]
    came_from = {}
    g_score = {source: 0}
    visited = set()
    explored_order = []

    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        explored_order.append(current)

        if current == target:
            path, node = [], current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(source)
            return path[::-1], g_score[target], explored_order

        for nc, nr, move_cost in get_walkable_neighbors(current[0], current[1], grid):
            neighbor = (nc, nr)
            if neighbor in visited:
                continue
            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, target)
                heapq.heappush(open_set, (f_score, next(counter), neighbor))

    return [], float('inf'), explored_order


class AStarSimulator:
    GRID_ROWS = 20
    GRID_COLS = 20
    WALL_VALUE = -1
    EMPTY_VALUE = 1

    def __init__(self):
        if 's' in plt.rcParams['keymap.save']:
            plt.rcParams['keymap.save'].remove('s')

        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(left=0.05, bottom=0.25, right=0.85)
        self.fig.canvas.manager.set_window_title("A* Pathfinding Simulator — 20×20 Grid")

        self.grid = np.ones((self.GRID_ROWS, self.GRID_COLS))
        self.start_node = (2, 2)
        self.goal_node = (17, 17)

        self.animation = None
        self.explored_nodes = []
        self.solution_path = []
        self.animation_step = 0
        self.path_cost = 0

        self._build_colormap()
        self._init_artists()
        self._init_controls()
        self.redraw()

        self.fig.canvas.mpl_connect('button_press_event',  self.on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_drag)
        self.fig.canvas.mpl_connect('key_press_event',     self.on_key_press)

    def _build_colormap(self):
        base = plt.get_cmap('turbo')
        colors = base(np.linspace(0, 1, 100))
        colors[0] = [1.0, 1.0, 1.0, 1.0]
        self.cell_cmap = mcolors.ListedColormap(colors)
        self.cell_cmap.set_under('black')
        self.cell_norm = plt.Normalize(vmin=1, vmax=100)

    def _init_artists(self):
        self.grid_image = self.ax.imshow(
            self.grid,
            cmap=self.cell_cmap,
            norm=self.cell_norm,
            interpolation='nearest',
            origin='upper'
        )

        self.start_marker    = self.ax.scatter([], [], c='lime',   marker='s', edgecolors='black', zorder=5)
        self.goal_marker     = self.ax.scatter([], [], c='red',    marker='s', edgecolors='black', zorder=5)
        self.explored_marker = self.ax.scatter([], [], c='gray',   alpha=0.5,  edgecolors='black', zorder=2)
        self.frontier_marker = self.ax.scatter([], [], c='orange', zorder=3)
        self.path_line,      = self.ax.plot([], [], c='magenta', linewidth=4, zorder=4)

    def _init_controls(self):
        self.ax_run_btn   = plt.axes([0.05, 0.1, 0.1, 0.04])
        self.run_btn      = Button(self.ax_run_btn,   'Run A*',    color='lightgreen')
        self.run_btn.on_clicked(self.run_pathfinding)

        self.ax_reset_btn = plt.axes([0.16, 0.1, 0.1, 0.04])
        self.reset_btn    = Button(self.ax_reset_btn, 'Reset Vis')
        self.reset_btn.on_clicked(self.reset_visualization)

        self.ax_clear_btn = plt.axes([0.27, 0.1, 0.1, 0.04])
        self.clear_btn    = Button(self.ax_clear_btn, 'Clear Map', color='lightcoral')
        self.clear_btn.on_clicked(self.clear_map)

        self.ax_speed_slider  = plt.axes([0.55, 0.13, 0.25, 0.03])
        self.speed_slider     = Slider(self.ax_speed_slider,  'Speed',   1, 100, valinit=10, valstep=1)

        self.ax_weight_slider = plt.axes([0.55, 0.08, 0.25, 0.03])
        self.weight_slider    = Slider(self.ax_weight_slider, 'Brush W', 2, 100, valinit=50, valstep=1, color='cyan')

        plt.figtext(
            0.86, 0.5,
            "Controls:\nL-Click: Wall\nR-Click: Clear\nS: Set Start\nG: Set Goal\nW: Paint Weight\n  (from Brush Slider)",
            fontsize=10,
            bbox=dict(facecolor='wheat', alpha=0.5)
        )

    def redraw(self):
        self.grid_image.set_data(self.grid)
        self.grid_image.set_extent([-0.5, self.GRID_COLS - 0.5, self.GRID_ROWS - 0.5, -0.5])

        marker_size = 50
        self.start_marker.set_sizes([marker_size * 5])
        self.goal_marker.set_sizes([marker_size * 5])
        self.explored_marker.set_sizes([marker_size])
        self.frontier_marker.set_sizes([marker_size * 3])

        self.ax.set_xticks(np.arange(-0.5, self.GRID_COLS + 0.5, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.GRID_ROWS + 0.5, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.5)
        self.ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        self.start_marker.set_offsets([self.start_node])
        self.goal_marker.set_offsets([self.goal_node])

        self.ax.set_xlim(-0.5, self.GRID_COLS - 0.5)
        self.ax.set_ylim(self.GRID_ROWS - 0.5, -0.5)
        self.ax.set_aspect('equal', adjustable='box')

        self.fig.canvas.draw_idle()

    def reset_visualization(self, event=None):
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None

        self.explored_marker.set_offsets(np.empty((0, 2)))
        self.frontier_marker.set_offsets(np.empty((0, 2)))
        self.path_line.set_data([], [])

        self.ax.set_title("A* Simulator: Ready (20×20)")
        self.fig.canvas.draw_idle()

    def _cell_from_event(self, event):
        if event.inaxes != self.ax:
            return None, None
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if 0 <= col < self.GRID_COLS and 0 <= row < self.GRID_ROWS:
            return col, row
        return None, None

    def paint_cell(self, event):
        col, row = self._cell_from_event(event)
        if col is None:
            return
        if event.button == 1:
            self.grid[row, col] = self.WALL_VALUE
        elif event.button == 3:
            self.grid[row, col] = self.EMPTY_VALUE
        self.reset_visualization()
        self.grid_image.set_data(self.grid)
        self.fig.canvas.draw_idle()

    def on_mouse_press(self, event):
        self.paint_cell(event)

    def on_mouse_drag(self, event):
        if event.button in [1, 3]:
            self.paint_cell(event)

    def on_key_press(self, event):
        col, row = self._cell_from_event(event)
        if col is None:
            return
        if event.key == 's':
            self.start_node = (col, row)
        elif event.key == 'g':
            self.goal_node = (col, row)
        elif event.key == 'w':
            self.grid[row, col] = int(self.weight_slider.val)
        self.reset_visualization()
        self.redraw()

    def clear_map(self, event):
        self.grid = np.ones((self.GRID_ROWS, self.GRID_COLS))
        self.grid_image.set_data(self.grid)
        self.reset_visualization()

    def run_pathfinding(self, event):
        self.reset_visualization()
        self.solution_path, self.path_cost, self.explored_nodes = run_astar(
            self.grid.tolist(), self.start_node, self.goal_node
        )

        if not self.explored_nodes:
            self.ax.set_title("No path found — start or goal is blocked!")
            return

        self.animation_step = 0
        self.animation = animation.FuncAnimation(
            self.fig, self._animation_tick,
            interval=30, blit=False, repeat=False, cache_frame_data=False
        )

    def _animation_tick(self, frame):
        batch_size = int(self.speed_slider.val)

        if self.animation_step < len(self.explored_nodes):
            end = min(self.animation_step + batch_size, len(self.explored_nodes))
            self.explored_marker.set_offsets(self.explored_nodes[:end])
            self.frontier_marker.set_offsets([self.explored_nodes[end - 1]])
            self.animation_step = end
            self.ax.set_title(f"Exploring… {self.animation_step} nodes visited")
        else:
            self.frontier_marker.set_offsets(np.empty((0, 2)))
            if self.solution_path:
                px, py = zip(*self.solution_path)
                self.path_line.set_data(px, py)
                self.ax.set_title(f"Done — cost: {self.path_cost}  |  nodes explored: {len(self.explored_nodes)}")
            else:
                self.ax.set_title(f"No path found — nodes explored: {len(self.explored_nodes)}")
            self.animation.event_source.stop()


if __name__ == "__main__":
    app = AStarSimulator()
    plt.show()
