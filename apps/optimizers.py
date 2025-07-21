import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(grad_fn1, grad_fn2, mo, toy_function1, toy_function2):
    dropdown_dict = mo.ui.dropdown(
        options={
            "Nice function": (toy_function1, grad_fn1, (-10, 10)), 
            "Bumpy function": (toy_function2, grad_fn2, (1, 3)), 
        },
        value="Nice function",
        label="Pick a function"
    )
    dropdown_dict
    return (dropdown_dict,)


@app.cell(hide_code=True)
def _(mo):
    lr_slider = mo.ui.slider(start=0.01, stop=10, step=0.01, label="Step size", show_value=True)
    num_iters_slider = mo.ui.slider(start=10, stop=1000, step=1, label="Number of iterations", show_value=True)
    x_init_ui = mo.ui.slider(start=-10, stop=10, step=0.1, value=0.0, label="Initial value", show_value=True)

    hparams = mo.ui.dictionary({
        "num_iters": num_iters_slider, 
        "lr": lr_slider,
        "x_init": x_init_ui
    }
    )
    hparams.vstack()
    return (hparams,)


@app.cell(hide_code=True)
def _(f_iterations, mo, np, x_grid, x_iterations, y_grid):
    import matplotlib
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    norm = mcolors.Normalize(vmin=0, vmax=len(x_iterations) - 1)
    cmap = matplotlib.colormaps["jet"]
    colors = [mcolors.to_hex(cmap(norm(i))) for i in range(len(x_iterations))]

    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.05, subplot_titles=("Algorithm 1", "Algorithm 2", "Algorithm 3"))
    for col in range(1, 4):
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_grid,
                mode="lines",
                name=f"Plot name {col}",
                line=dict(color="black"),
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_iterations,
                y=f_iterations,
                mode="markers+lines",
                marker=dict(
                    size=8,
                    color=colors
                ),
                line=dict(color="red", width=0.3),
                showlegend=False
           ),
            row=1,
            col=col
        )
    for col in range(1, 4):
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(f_iterations)),
                y=f_iterations,
                mode="lines+markers",
                name=f"Plot name {col}",
                line=dict(color="black"),
                marker=dict(size=4),
                showlegend=False,
            ),
            row=2,
            col=col,
        )
    fig.update_layout(height=600, width=1000, title_text="Optimization algorithms")

    plot = mo.ui.plotly(fig)
    plot
    return


@app.cell(hide_code=True)
def _():
    import numpy as np
    from numpy.typing import NDArray

    def toy_function1(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.log(1.0 + np.exp(x)) + 0.1*x**2

    def grad_fn1(x: NDArray[np.float64]) -> NDArray[np.float64]:
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid + 0.2 * x

    a = 1.0
    def toy_function2(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (x-2)**2 + 0.3*np.sin(10.0*x)

    def grad_fn2(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2.0*(x-2) + 3.0*np.cos(10.0*x)

    def toy_function3(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x**6 - 6*x**4 + 9*x**2 + 0.5 * x

    def grad_fn3(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 6*x**5 - 24*x**3 + 18*x + 0.5
    return NDArray, grad_fn1, grad_fn2, np, toy_function1, toy_function2


@app.cell(hide_code=True)
def _(NDArray, dropdown_dict, hparams, np):

    objective_fn = dropdown_dict.value[0]
    grad_fn = dropdown_dict.value[1]
    bounds = dropdown_dict.value[2]

    def gradient_descent(n_iters: int, step_size: float, x_init: NDArray[np.float64]) -> NDArray[np.float64]:
        x = x_init.copy()
        stacked_values = [x]
        for _ in range(n_iters):
            grad = grad_fn(x)
            x = x - step_size * grad
            stacked_values.append(x)
        return np.array(stacked_values)

    x_init = np.array([hparams["x_init"].value])
    x_iterations = gradient_descent(hparams["num_iters"].value, hparams["lr"].value, x_init).squeeze()
    f_iterations = np.array([objective_fn(xi) for xi in x_iterations]).squeeze()

    x_grid = np.linspace(np.minimum(bounds[0], np.min(x_iterations)), np.maximum(bounds[1], np.max(x_iterations)), 1000)
    y_grid = objective_fn(x_grid)

    return f_iterations, x_grid, x_iterations, y_grid


if __name__ == "__main__":
    app.run()
