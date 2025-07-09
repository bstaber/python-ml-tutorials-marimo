import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ridge Regression

    This notebook demonstrates Ridge regression, a regularized version of linear regression that helps prevent overfitting and improves model stability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1D Ridge Regression
    
    We begin with a simple 1D case. The model is given by $y = wx + b + \epsilon$ where $\epsilon$ is Gaussian noise. The goal is to learn the parameters $w$ and $b$ from the data. 
    
    Ridge regression solves: 
    
    \[
        \min_{w, b} \sum_{i=1}^N (y_i - wx_i - b)^2 + \alpha w^2
    \]

    The regularization term $\alpha w^2$ penalizes large weights, improving stability when data is noisy or features are correlated. Use the sliders to play with the sample size, noise level, and the Ridge regularization strength.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    slider_N = mo.ui.slider(
        start=100, stop=1000, step=1, show_value=True, label="Sample size N"
    )
    slider_noise = mo.ui.slider(
        start=0, stop=4.0, step=1e-4, show_value=True, label="Noise level"
    )
    slider_alpha = mo.ui.slider(
        start=0, stop=1000, step=1.0, show_value=True, label="Ridge regularization"
    )

    hparams = mo.ui.dictionary(
        {"N": slider_N, "epsilon": slider_noise, "alpha": slider_alpha}
    )
    hparams.vstack()
    return (hparams,)


@app.cell(hide_code=True)
def _(hparams, np, x, y):
    from numpy.typing import NDArray

    def ridge_regression_1d(
        x: NDArray[np.float64], y: NDArray[np.float64], alpha: float
    ):
        """Perform 1D Ridge regression."""
        X = np.vstack([x, np.ones_like(x)]).T

        # Ridge solution: w = (X^T X + alpha * I)^(-1) X^T y
        I = np.eye(X.shape[1])
        I[1, 1] = 0

        w_ridge = np.linalg.inv(X.T @ X + alpha * I) @ X.T @ y
        w, b = w_ridge[0], w_ridge[1]
        return w, b

    w, b = ridge_regression_1d(x, y, alpha=hparams["alpha"].value)
    y_prediction = w * x + b
    return NDArray, y_prediction


@app.cell(hide_code=True)
def _(hparams):
    import numpy as np

    x = 6 * np.random.rand(hparams["N"].value) - 3
    y = 2.0 * x + 1.0 + hparams["epsilon"].value * np.random.randn(hparams["N"].value)
    return np, x, y


@app.cell(hide_code=True)
def _(mo, x, y, y_prediction):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="Training data", marker=dict(color="blue")
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_prediction,
            mode="lines",
            name="Ridge prediction",
            line=dict(color="red"),
        )
    )
    plot = mo.ui.plotly(fig)
    plot
    return (go,)


@app.cell
def _(mo):
    mo.md(
        """In this case, Ridge regression might not bring significant benefits over OLS, as the data is relatively simple and linear. However, it sets the stage for understanding how Ridge can help in more complex scenarios."""
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Ridge Regression in Multiple Dimensions

    In higher dimensions, we want to fit a linear model of the form:

    \[
    y = X \mathbf{w} + b + \varepsilon
    \]

    where:
    
    - $X \in \mathbb{R}^{n \times d}$ is the input matrix (with $n$ samples and $d$ features),
    - $\mathbf{w} \in \mathbb{R}^d$ is the weight vector,
    - $y \in \mathbb{R}^n$ is the target vector,
    - $\varepsilon$ is a noise term.

    **Ordinary Least Squares (OLS)** estimates $\mathbf{w}$ by minimizing the residual sum of squares: $\min_{\mathbf{w}} \| y - X \mathbf{w} \|^2$. However, if the features are **colinear** or if $d > n$, the matrix $X^T X$ becomes ill-conditioned or even singular. **Ridge regression** addresses this by adding a regularization term:

    \[
    \min_{\mathbf{w}} \| y - X \mathbf{w} \|^2 + \alpha \| \mathbf{w} \|^2
    \]

    The solution becomes:

    \[
    \mathbf{w}_{\text{ridge}} = (X^T X + \alpha I)^{-1} X^T y
    \]

    This regularization has several benefits that we will explore. When $\alpha = 0$, Ridge reduces to OLS. When $\alpha$ is large, coefficients are heavily penalized.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Ridge for colinear features

    Ridge regression is particularly useful when input features are colinear, i.e., linearly dependent.

    In ordinary least squares (OLS), colinearity leads to instability in the solution. Ridge adds a penalty on weight magnitude, which stabilizes the solution.

    We simulate two features $x_1$ and $x_2$ that are colinear:
    
    - $x_2 = (1 - \rho) x_1 + \rho \cdot \text{noise}$
    - Adjusting  $\rho \in [0, 1]$ controls colinearity
    - $\rho \approx 0$: high colinearity  
    - $\rho \approx 1$: nearly independent features

    Use the slider to adjust the colinearity factor $\rho$. Then observe how the learned coefficients $w_0, w_1$ change with $\alpha$.
    """)
    return


@app.cell(hide_code=True)
def _(NDArray, np):
    def ridge_regression_nd(
        X: NDArray[np.float64], y: NDArray[np.float64], alpha: float
    ):
        X_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        X_centered = X - X_mean
        y_centered = y - y_mean

        n_features = X.shape[1]
        A = X_centered.T @ X_centered + alpha * np.eye(n_features)
        w = np.linalg.solve(A, X_centered.T @ y_centered)

        # Recover intercept
        b = y_mean - X_mean @ w

        return w, b

    return (ridge_regression_nd,)


@app.cell(hide_code=True)
def _(mo):
    slider_colinearity = mo.ui.slider(
        start=1e-6, stop=1.0, step=1e-4, show_value=True, label="Colinearity factor"
    )
    slider_colinearity
    return (slider_colinearity,)


@app.cell(hide_code=True)
def _(slider_colinearity, np):
    np.random.seed(0)
    n_samples = 1000
    x1 = np.random.rand(n_samples)
    x2 = (
        1 - slider_colinearity.value
    ) * x1 + slider_colinearity.value * np.random.rand(n_samples)

    X = np.vstack([x1, x2]).T
    true_w = np.array([3.0, 3.0])
    y_2d = X @ true_w + np.random.randn(n_samples)
    return X, y_2d


@app.cell(hide_code=True)
def _(X, go, mo, np, ridge_regression_nd, y_2d):
    lambdas = np.logspace(-15, 20, 100)
    weights = []

    for lam in lambdas:
        w_2d, _ = ridge_regression_nd(X, y_2d, lam)
        weights.append(w_2d)

    weights = np.array(weights)

    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=np.log10(lambdas), y=weights[:, 0], name="w[0]"))
    fig2d.add_trace(go.Scatter(x=np.log10(lambdas), y=weights[:, 1], name="w[1]"))
    fig2d.update_layout(
        title="Ridge coefficients vs log10(alpha)",
        xaxis_title="log10(alpha)",
        yaxis_title="weights",
        width=800,
        height=400,
        legend=dict(
            title="",
            x=10,
            y=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font_size=18,
        ),
    )
    plot2d = mo.ui.plotly(fig2d)
    plot2d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing Ridge Regression in 2D

    Now we visualize the effect of Ridge regression in a 2D space. We will fit a surface to the data points and see how the regularization parameter $\alpha$ affects the model. Use the slider to adjust $\alpha$ and observe how the fitted surface changes.
    
    You can also change the colinearity factor to see how it affects the stability of the solution. The predicted surface is compared against the true response surface. Is it close to the true response?
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    slider_alpha_2d = mo.ui.slider(
        start=1e-15, stop=10.0, step=1e-6, show_value=True, label="Ridge regularization"
    )
    slider_alpha_2d
    return (slider_alpha_2d,)


@app.cell(hide_code=True)
def _(X, go, slider_alpha_2d, mo, np, ridge_regression_nd, y_2d):
    xx1, xx2 = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    X_grid = np.vstack([xx1.ravel(), xx2.ravel()]).T

    w_ridge, b_ridge = ridge_regression_nd(X, y_2d, slider_alpha_2d.value)
    y_pred_grid = (X_grid @ w_ridge + b_ridge).reshape(xx1.shape)
    y_true_grid = 3.0 * xx1 + 3.0 * xx2

    fig_surf = go.Figure()
    fig_surf.add_trace(
        go.Surface(
            z=y_pred_grid,
            x=xx1,
            y=xx2,
            colorscale="Blues",
            name="Prediction",
            showlegend=True,
        )
    )
    fig_surf.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=y_2d,
            mode="markers",
            marker=dict(size=4, color="black"),
            name="Training points",
            showlegend=True,
        )
    )
    fig_surf.add_trace(
        go.Surface(
            z=y_true_grid,
            x=xx1,
            y=xx2,
            colorscale="Reds",
            opacity=1.0,
            showscale=False,
            name="True response",
            showlegend=True,
        )
    )
    fig_surf.update_layout(
        xaxis_title="x1",
        yaxis_title="x2",
        width=800,
        height=600,
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="z",
            camera=dict(eye=dict(x=0, y=-1.5, z=0.8)),
        ),
        legend=dict(
            title="",
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="black",
            borderwidth=1,
            font_size=18,
        ),
    )
    plot_surf = mo.ui.plotly(fig_surf)
    plot_surf
    return


@app.cell(hide_code=True)
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More features than samples

    Ridge regression is also effective when the number of features $d$ exceeds the number of samples $n$. This is called the high-dimensional regime.

    In this case OLS fails because $X^T X$ is not invertible. Ridge adds $\alpha I$, making the matrix invertible and the solution well-defined.

    Use the sliders to increase the number of input features and tune $\alpha$. When do you start to see the benefits of Ridge regression?
    """)

    return


@app.cell(hide_code=True)
def _(hparams_nd, np):
    np.random.seed(0)
    n_samples_nd = 20
    n_features = int(hparams_nd["num_features"].value)  # More features than samples

    X_nd = np.random.randn(n_samples_nd, n_features)
    true_w_nd = np.random.randn(n_features)
    y_nd = X_nd @ true_w_nd + np.random.randn(n_samples_nd) * 0.1
    return X_nd, y_nd


@app.cell(hide_code=True)
def _(mo):
    slider_num_features = mo.ui.slider(
        start=2, stop=100, step=1.0, show_value=True, label="Num. features"
    )
    slider_alpha_nd = mo.ui.slider(
        start=0, stop=2.0, step=1e-4, show_value=True, label="Ridge regularization"
    )

    hparams_nd = mo.ui.dictionary(
        {"num_features": slider_num_features, "alpha": slider_alpha_nd}
    )
    hparams_nd.vstack()
    return (hparams_nd,)


@app.cell(hide_code=True)
def _(X_nd, hparams_nd, ridge_regression_nd, y_nd):
    w_ridge_nd, b_ridge_nd = ridge_regression_nd(X_nd, y_nd, hparams_nd["alpha"].value)
    w_ols_nd, b_ols_nd = ridge_regression_nd(X_nd, y_nd, 0.0)
    return w_ols_nd, w_ridge_nd


@app.cell(hide_code=True)
def _(go, hparams_nd, np, w_ols_nd, w_ridge_nd):
    features = np.arange(hparams_nd["num_features"].value)

    fig_nd = go.Figure()
    fig_nd.add_trace(
        go.Scatter(x=features, y=w_ols_nd, name="OLS", marker_color="red", opacity=0.7)
    )

    fig_nd.add_trace(
        go.Scatter(
            x=features, y=w_ridge_nd, name="Ridge", marker_color="blue", opacity=0.7
        )
    )

    fig_nd.update_layout(
        title="OLS vs Ridge coefficients",
        xaxis_title="Feature index",
        yaxis_title="Coefficient value",
        barmode="group",  # group bars side by side
        width=1000,
        height=500,
        legend=dict(x=10, y=0.99),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preventing Overfitting with Ridge

    Overfitting occurs when a model is too flexible relative to the amount of data. To illustrate this, we will fit a polynomial regression model to a 1D dataset. High-degree polynomials can perfectly fit noisy data, leading to poor generalization.

    We compare OLS polynomial regression ($\alpha = 0$) and polynomial Ridge regression with ($\alpha > 0$).

    Use the sliders to change the polynomial degree and adjust $\alpha$. Observe when does OLS start overfitting and how does Ridge help control model complexity?
    """)
    return


@app.cell(hide_code=True)
def _(np):
    np.random.seed(42)

    n_samples_x_poly = 40
    x_poly = np.sort(np.random.rand(n_samples_x_poly))
    y_poly = np.cos(2.0 * np.pi * x_poly) + 0.5 * np.random.randn(n_samples_x_poly)

    def design_matrix(x, degree):
        return np.vstack([x**d for d in range(degree + 1)]).T

    return design_matrix, x_poly, y_poly


@app.cell(hide_code=True)
def _(design_matrix, hparams_poly, np, x_poly, y_poly):
    X_poly = design_matrix(x_poly, hparams_poly["degree"].value)
    w_ols_poly = np.linalg.solve(X_poly.T @ X_poly, X_poly.T @ y_poly)

    I = np.eye(X_poly.shape[1])
    w_ridge_poly = np.linalg.solve(
        X_poly.T @ X_poly + hparams_poly["alpha"].value * I, X_poly.T @ y_poly
    )
    return w_ols_poly, w_ridge_poly


@app.cell(hide_code=True)
def _(mo):
    slider_degree = mo.ui.slider(
        start=2,
        stop=30,
        step=1,
        show_value=True,
        label="Polynomial degree",
    )
    slider_alpha_poly = mo.ui.slider(
        start=1e-15,
        stop=1.0,
        step=1e-4,
        show_value=True,
        label="Ridge regularization",
    )

    hparams_poly = mo.ui.dictionary(
        {"degree": slider_degree, "alpha": slider_alpha_poly}
    )
    hparams_poly.vstack()
    return (hparams_poly,)


@app.cell(hide_code=True)
def _(
    design_matrix,
    go,
    hparams_poly,
    np,
    w_ols_poly,
    w_ridge_poly,
    x_poly,
    y_poly,
):
    x_plot = np.linspace(0.1, 0.9, 300)
    X_plot = design_matrix(x_plot, hparams_poly["degree"].value)
    y_pred_ols = X_plot @ w_ols_poly
    y_pred_ridge = X_plot @ w_ridge_poly

    fig_poly = go.Figure()
    fig_poly.add_trace(
        go.Scatter(
            x=x_poly,
            y=y_poly,
            mode="markers",
            name="Training data",
            marker=dict(color="black"),
        )
    )
    fig_poly.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_pred_ols,
            mode="lines",
            name="OLS (overfit)",
            line=dict(color="red"),
        )
    )
    fig_poly.add_trace(
        go.Scatter(
            x=x_plot,
            y=y_pred_ridge,
            mode="lines",
            name="Ridge (regularized)",
            line=dict(color="blue"),
        )
    )
    fig_poly.update_layout(
        title=f"Polynomial degree {hparams_poly['degree'].value}",
        xaxis_title="x",
        yaxis_title="y",
        width=800,
        height=500,
        legend=dict(x=10, y=0.99),
    )
    return


if __name__ == "__main__":
    app.run()
