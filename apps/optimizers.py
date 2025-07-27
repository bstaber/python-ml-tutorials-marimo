import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(grad_fn1, grad_fn2, mo, toy_function1, toy_function2):
    dropdown_dict = mo.ui.dropdown(
        options={
            "Convex function": (toy_function1, grad_fn1, (-10, 10)), 
            "Non convex function": (toy_function2, grad_fn2, (1, 3)), 
        },
        value="Convex function",
        label="Pick a function"
    )
    dropdown_dict
    return (dropdown_dict,)


@app.cell(hide_code=True)
def _(mo):
    lr_slider = mo.ui.slider(start=0.001, stop=10, step=0.001, label="Step size", show_value=True)
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
def _(
    f_iterations_gd,
    f_iterations_momentum,
    f_iterations_nag,
    mo,
    np,
    x_grid,
    x_iterations_gd,
    x_iterations_momentum,
    x_iterations_nag,
    y_grid,
):
    import matplotlib
    import matplotlib.colors as mcolors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    norm = mcolors.Normalize(vmin=0, vmax=len(x_iterations_gd) - 1)
    cmap = matplotlib.colormaps["jet"]
    colors = [mcolors.to_hex(cmap(norm(i))) for i in range(len(x_iterations_gd))]

    def add_trace_algorithm(fig, x_iterations, f_iterations, row, col):
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
            row=row,
            col=col
        )

    fig = make_subplots(
        rows=2, 
        cols=3, 
        vertical_spacing=0.15, 
        specs=[
            [{}, {}, {}],                # Row 1: 3 separate plots
            [{"colspan": 3}, None, None]  # Row 2: 1 plot spanning all 3 columns
        ], 
        subplot_titles=("GD", "GD + momentum", "Nesterov AG")
    )

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

    add_trace_algorithm(fig, x_iterations_gd, f_iterations_gd, 1, 1)
    add_trace_algorithm(fig, x_iterations_momentum, f_iterations_momentum, 1, 2)
    add_trace_algorithm(fig, x_iterations_nag, f_iterations_nag, 1, 3)

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_iterations_gd)),
            y=f_iterations_gd,
            mode="lines+markers",
            name="Gradient descent",
            line=dict(color="black"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_iterations_momentum)),
            y=f_iterations_momentum,
            mode="lines+markers",
            name="Gradient descent + momentum",
            line=dict(color="red"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(f_iterations_nag)),
            y=f_iterations_nag,
            mode="lines+markers",
            name="NAG",
            line=dict(color="blue"),
            marker=dict(size=4),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=600, 
        width=1000, 
        title_text="Optimization algorithms", 
        legend=dict(
            x=1.02,
            y=0.25,
            xanchor="left",
            yanchor="middle"
        ))
    fig.update_xaxes(title_text="Iterations", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    for i in range(3):
        fig.update_xaxes(title_text="x", row=1, col=i+1)
        fig.update_yaxes(title_text="f(x)", row=1, col=i+1)

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

    def gradient_descent_with_momentum(
        n_iters: int,
        step_size: float,
        momentum: float,
        x_init: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x = x_init.copy()
        v = np.zeros_like(x)
        stacked_values = [x]
        for _ in range(n_iters):
            grad = grad_fn(x)
            v = momentum * v - step_size * grad
            x = x + v
            stacked_values.append(x)
        return np.array(stacked_values)

    def nesterov_accelerated_gradient(
        n_iters: int,                            
        step_size: float,           
        x_init: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        x_prev = x_init.copy()
        y = x_init.copy()
        t_prev = 1.0

        trajectory = [x_init]

        for _ in range(n_iters):
            grad = grad_fn(y)
            x_next = y - step_size * grad
            t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_prev**2))
            beta = (t_prev - 1) / t_next
            y = x_next + beta * (x_next - x_prev)

            trajectory.append(x_next)

            x_prev = x_next
            t_prev = t_next

        return np.array(trajectory)

    x_init = np.array([hparams["x_init"].value])

    x_iterations_gd = gradient_descent(hparams["num_iters"].value, hparams["lr"].value, x_init).squeeze()
    f_iterations_gd = np.array([objective_fn(xi) for xi in x_iterations_gd]).squeeze()

    x_iterations_momentum = gradient_descent_with_momentum(hparams["num_iters"].value, hparams["lr"].value, 0.95, x_init).squeeze()
    f_iterations_momentum = np.array([objective_fn(xi) for xi in x_iterations_momentum]).squeeze()

    x_iterations_nag = nesterov_accelerated_gradient(hparams["num_iters"].value, hparams["lr"].value, x_init).squeeze()
    f_iterations_nag = np.array([objective_fn(xi) for xi in x_iterations_nag]).squeeze()

    x_grid = np.linspace(np.minimum(bounds[0], np.min(x_iterations_gd)), np.maximum(bounds[1], np.max(x_iterations_gd)), 1000)
    y_grid = objective_fn(x_grid)
    return (
        f_iterations_gd,
        f_iterations_momentum,
        f_iterations_nag,
        x_grid,
        x_iterations_gd,
        x_iterations_momentum,
        x_iterations_nag,
        y_grid,
    )


if __name__ == "__main__":
    app.run()
