
# Install required libraries
%pip install numpy==1.26.4 pysindy pysr pandas sympy scipy scikit-learn --quiet

# Upload CSV file
from google.colab import files
uploaded = files.upload()

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load data
import pandas as pd
import numpy as np
filename = list(uploaded.keys())[0]
data = pd.read_csv(filename)
t = data.iloc[:, 0].values
x = data.iloc[:, 1].values
dt = t[1] - t[0]

# Save all outputs to text file
output_lines = []

output_lines.append("### Step 1: Symbolic Regression and Curve Fitting\n")

# --- Curve fitting (Polynomial) ---
from scipy.optimize import curve_fit
def poly3(t, a, b, c, d): return a*t**3 + b*t**2 + c*t + d
params, _ = curve_fit(poly3, t, x)
poly_expr = f"{params[0]:.4f}*t**3 + {params[1]:.4f}*t**2 + {params[2]:.4f}*t + {params[3]:.4f}"
output_lines.append(f"Polynomial Curve Fit: x(t) ≈ {poly_expr}\n")

# --- PySR symbolic regression ---
from pysr import PySRRegressor
from sympy import symbols, Eq, sympify

try:
    model_pysr = PySRRegressor(
        niterations=40,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        verbosity=0
    )
    model_pysr.fit(t.reshape(-1, 1), x)
    rhs_pysr = sympify(str(model_pysr.sympy()))
    output_lines.append(f"PySR Regression: dx/dt ≈ {rhs_pysr}\n")
except Exception as e:
    rhs_pysr = None
    output_lines.append(f"PySR symbolic regression failed: {e}\n")

# --- Step 2: Sparse Identification of Nonlinear Dynamics ---
output_lines.append("### Step 2: SINDy Differential Equation Discovery\n")
print("### Choose method to discover differential equation:")
print("1. STLSQ\n2. SR3\n3. Lasso\n4. ElasticNet\n5. EnsembleSINDy\n6. PySR")
method = int(input("Enter choice (1–6): "))
rhs_expr = None

if method < 6:
    import pysindy as ps
    from sklearn.linear_model import Lasso, ElasticNet

    if method == 1:
        optimizer = ps.STLSQ()
    elif method == 2:
        optimizer = ps.SR3()
    elif method == 3:
        optimizer = ps.SINDyOptimizer(Lasso(alpha=0.01))
    elif method == 4:
        optimizer = ps.SINDyOptimizer(ElasticNet(alpha=0.01, l1_ratio=0.8))
    elif method == 5:
        optimizer = ps.EnsembleOptimizer([ps.STLSQ(), ps.SR3()])

    poly_lib = ps.PolynomialLibrary(degree=3)
    model = ps.SINDy(optimizer=optimizer, feature_library=poly_lib)
    model.fit(x.reshape(-1, 1), t=t)
    feature_names = model.feature_library.get_feature_names(["x"])
    coef = model.coefficients()[0]
    x_sym = symbols("x")
    rhs_expr = sum(c * sympify(name) for c, name in zip(coef, feature_names) if abs(c) > 1e-6)
    output_lines.append(f"SINDy Equation: dx/dt ≈ {rhs_expr}\n")
else:
    rhs_expr = rhs_pysr  # fallback to PySR

# Solve equation
from sympy import dsolve, Function
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp

if rhs_expr is not None:
    print("\n### Choose method to solve the equation:")
    print("1. Symbolic Integration (SymPy)")
    print("2. Numerical Integration (SciPy)")
    solver_choice = int(input("Enter choice (1 or 2): "))
    t_sym, x_sym = symbols("t x")

    if solver_choice == 1:
        x_func = Function("x")
        dxdt = Eq(x_func(t_sym).diff(t_sym), rhs_expr)
        sol = dsolve(dxdt, x_func(t_sym))
        output_lines.append("\n### Symbolic Integration Solution:\n")
        output_lines.append(str(sol) + "\n")
    elif solver_choice == 2:
        f = lambdify(x_sym, rhs_expr, modules=["numpy"])
        def ode(t, y): return f(y)
        sol = solve_ivp(ode, [t[0], t[-1]], [x[0]], t_eval=t)
        output_lines.append("\n### Numerical Integration Result (first 5 points):\n")
        output_lines.append(f"t: {sol.t[:5]}\n")
        output_lines.append(f"x: {sol.y[0][:5]}\n")
else:
    output_lines.append("Unable to solve due to symbolic parsing failure.\n")

# Suggestions
output_lines.append("\n### Recommendations:\n")
output_lines.append("- Use STLSQ or SR3 for sparse models.\n")
output_lines.append("- EnsembleSINDy increases robustness by combining methods.\n")
output_lines.append("- Use PySR for accurate symbolic expressions.\n")
output_lines.append("- Symbolic solutions are elegant but only work for integrable forms.\n")
output_lines.append("- Use SciPy’s solve_ivp for stiff/non-stiff ODEs.\n")

# Write outputs to file
with open("discovered_equations.txt", "w") as f:
    f.write("\n".join(output_lines))
print("Equations and suggestions written to 'discovered_equations.txt'")

# Create requirements.txt
with open("requirements.txt", "w") as f:
    f.write("python>=3.8\n")
    f.write("numpy==1.26.4\n")
    f.write("pandas\n")
    f.write("scipy\n")
    f.write("sympy\n")
    f.write("pysindy\n")
    f.write("pysr\n")
    f.write("scikit-learn\n")
print("Requirements written to 'requirements.txt'")

# Automatically trigger download
from google.colab import files
files.download("discovered_equations.txt")
files.download("requirements.txt")
