using CairoMakie
using LinearAlgebra
using QuadGK

nx, ny = 10, 40
Lx, Ly = 1.0, 1.0
x0, y0 = 0.0, 0.0
dx, dy = Lx/nx, Ly/ny

x = collect(0:dx:Lx)
xc = 0.5*(x[1:end-1] + x[2:end])

# Define the vector of Heights
H = rand(length(xc))

# Interpolation linéaire
function height_interpolation_linear(x_mesh, H_values)
    nx = length(H_values)
    N = 2*nx
    
    # Points aux interfaces et aux centres
    x = x_mesh
    xc = 0.5 * (x[1:end-1] + x[2:end])
    dx = x[2] - x[1]
    
    # Valeurs aux interfaces (à déterminer)
    M = zeros(N, N)
    b = zeros(N)

    function idx(i, locale)
        return 2*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i * dx = ∫(ai + bi*x) dx from 0 to dx
        #    This expands to: H_i * dx = ai*dx + bi*dx^2/2
        #    Dividing by dx: H_i = ai + bi*dx/2
        rowV = 2*(i-1) + 1
        M[rowV, idx(i,0)] = 1.0        # a_i
        M[rowV, idx(i,1)] = 0.5*dx     # b_i * dx/2
        b[rowV] = H_values[i]

        # 2) Value continuity eq : a_i + b_i*dx = a_{i+1}
        rowC = 2*(i-1) + 2
        M[rowC, idx(i,0)] = 1.0      # a_i
        M[rowC, idx(i,1)] = dx       # b_i * dx
        M[rowC, idx(ip1,0)] = -1.0   # -a_{i+1}
        b[rowC] = 0.0
    end

    # We have N-1 equations so far, we need one more for periodicity
    # Replace one of the continuity equations with periodicity: b_1 = b_{nx}
    M[N, idx(1,1)] = 1.0
    M[N, idx(nx,1)] = -1.0
    b[N] = 0.0

    s = M \ b

    a, bc = s[1:2:end], s[2:2:end]

    # Fonction d'interpolation
    function h_tilde(x_val)
        # Trouver dans quelle cellule se trouve x_val
        i = 1
        while i <= nx && !(x[i] <= x_val && x_val <= x[i+1])
            i += 1
        end
        
        if i > nx
            return 0.0  # En dehors du domaine
        end
        
        # Interpolation linéaire entre les interfaces
        xi = x_val - x[i]
        return a[i] + bc[i]*xi  # Simpler representation
    end

    return h_tilde, s
end


# Utilisation
h_interp_linear, interface_values = height_interpolation_linear(x, H)

# Vérification de la conservation du volume
function verify_conservation(h_interp, x, H, order=1)
    nx = length(H)
    errors = zeros(nx)
    
    for i in 1:nx
        # For higher-order polynomials, we need more integration points
        n_points = 1000 * order  # Scale with polynomial order
        dx_local = (x[i+1] - x[i]) / n_points
        integral = 0.0
        
        # Use Gauss-Legendre quadrature for higher accuracy
        if order > 1
            # Simple 10-point Gauss-Legendre quadrature for each subinterval
            points = [-0.9739065285171717, -0.8650633666889844, -0.6794095682990244, -0.4333953941292472, -0.14887433898163127, 0.14887433898163116, 0.4333953941292472, 0.6794095682990244, 0.8650633666889844, 0.9739065285171717]
            weights = [0.06667134430868811, 0.14945134915058084, 0.2190863625159821, 0.26926671930999635, 0.29552422471475276, 0.29552422471475276, 0.26926671930999635, 0.2190863625159821, 0.14945134915058084, 0.06667134430868811]
            
            for j in 1:n_points
                x_left = x[i] + (j-1) * dx_local
                for (pt, wt) in zip(points, weights)
                    x_val = x_left + 0.5*dx_local * (1 + pt)
                    integral += h_interp(x_val) * wt * dx_local * 0.5
                end
            end
        else
            # Simple trapezoid rule for linear
            for j in 1:n_points
                x_val = x[i] + (j-0.5) * dx_local
                integral += h_interp(x_val) * dx_local
            end
        end
        
        # Compare with expected value
        expected = H[i] * (x[i+1] - x[i])
        errors[i] = (integral - expected) / expected
    end
    
    return errors
end

# Tracé des résultats
x_fine = range(minimum(x), maximum(x), length=500)
h_vals_linear = [h_interp_linear(xi) for xi in x_fine]

fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Hauteur", title="Interpolation linéaire conservant le volume")
lines!(ax, x_fine, h_vals_linear, label="Interpolation linéaire h_tilde(x)", linewidth=2)
scatter!(ax, xc, H, label="Hauteurs originales H", markersize=6)
#scatter!(ax, x, interface_values, label="Valeurs aux interfaces", markersize=4, color=:green)
axislegend(ax)
display(fig)

# Afficher les erreurs de conservation
errors_linear = verify_conservation(h_interp_linear, x, H, 1)
println("Erreurs maximales de conservation (linéaire): ", maximum(abs.(errors_linear)))

# Interpolation Quadratique
function height_interpolation_quadratic(x_mesh, H)
    nx = length(H)
    N = 3*nx
    
    # Points aux interfaces et aux centres
    x = x_mesh
    xc = 0.5 * (x[1:end-1] + x[2:end])
    dx = x[2] - x[1]
    
    # Valeurs aux interfaces (à déterminer)
    # We need 3*nx equations for 3*nx unknowns
    # We have 3 equations per cell (nx cells), but need to replace two equations for BCs
    A = zeros(N, N)
    rhs = zeros(N)

    function idx(i, locale)
        return 3*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i = a_i + 0.5 Δx b_i + 1/3 Δx^2 c_i
        rowV = 3*(i-1) + 1
        A[rowV, idx(i,0)] = 1.0        # a_i
        A[rowV, idx(i,1)] = 0.5*dx     # b_i
        A[rowV, idx(i,2)] = 1/3*dx^2   # c_i
        rhs[rowV] = H[i]

        # 2) Value continuity eq : a_i + b_i Δx + c_i Δx^2 = a_{i+1}
        rowC1 = 3*(i-1) + 2
        A[rowC1, idx(i,0)] = 1.0      # a_i
        A[rowC1, idx(i,1)] = dx       # b_i
        A[rowC1, idx(i,2)] = dx^2     # c_i
        A[rowC1, idx(ip1,0)] = -1.0   # -a_{i+1}
        rhs[rowC1] = 0.0

        # 3) Derivative continuity eq : b_i + 2 c_i Δx = b_{i+1}
        rowC2 = 3*(i-1) + 3
        A[rowC2, idx(i,1)] = 1.0      # b_i
        A[rowC2, idx(i,2)] = 2.0*dx   # c_i
        A[rowC2, idx(ip1,1)] = -1.0   # -b_{i+1}
        rhs[rowC2] = 0.0
    end
    
    # 4) BC : a_1 = a_{n_x}, b_1 = b_{n_x}, c_1 = c_{n_x} (periodicity)
    # Replace the last two equations for boundary conditions
    A[N-1, idx(1,0)] = 1.0
    A[N-1, idx(nx,0)] = -1.0
    rhs[N-1] = 0.0

    A[N, idx(1,1)] = 1.0
    A[N, idx(nx,1)] = -1.0
    rhs[N] = 0.0

    s = A \ rhs

    a, b, c = s[1:3:end], s[2:3:end], s[3:3:end]

    # Fix the interpolation function in height_interpolation_quadratic
    function h_tilde(x_val)
        # Find which cell contains x_val
        i = 1
        while i <= nx && !(x[i] <= x_val && x_val <= x[i+1])
            i += 1
        end
        
        if i > nx
            return 0.0  # Outside domain
        end
        
        # Local coordinate within the cell
        xi = x_val - x[i]
        
        # Direct polynomial evaluation with coefficients a, b, c
        return a[i] + b[i]*xi + c[i]*xi^2
    end
    
    return h_tilde, s
end

# Utilisation
h_interp_quadratic, interface_values = height_interpolation_quadratic(x, H)

# Vérification de la conservation du volume
errors_quadratic = verify_conservation(h_interp_quadratic, x, H, 2)
println("Erreurs maximales de conservation (quadratique): ", maximum(abs.(errors_quadratic)))

# Tracé des résultats
h_vals_quadratic = [h_interp_quadratic(xi) for xi in x_fine]

fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Hauteur", title="Interpolation quadratique conservant le volume")
lines!(ax, x_fine, h_vals_quadratic, label="Interpolation quadratique h_tilde(x)", linewidth=2)
scatter!(ax, xc, H, label="Hauteurs originales H", markersize=6)
#scatter!(ax, x, interface_values, label="Valeurs aux interfaces", markersize=4, color=:green)
axislegend(ax)
display(fig)

# Interpolation Cubique
function height_interpolation_cubic(x_mesh, H)
    nx = length(H)
    N = 4*nx
    
    # Points aux interfaces
    x = x_mesh
    xc = 0.5 * (x[1:end-1] + x[2:end])
    dx = x[2] - x[1]

    # Valeurs aux interfaces (à déterminer)
    # We need 4*nx equations for 4*nx unknowns
    # We have 4 equations per cell (nx cells), but need to replace three equations for BCs
    A = zeros(N, N)
    rhs = zeros(N)

    function idx(i, locale)
        return 4*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i = a_i + 0.5 Δx b_i + 1/3 Δx^2 c_i + 0.25 Δx^3 d_i
        rowV = 4*(i-1) + 1
        A[rowV, idx(i,0)] = 1.0        # a_i
        A[rowV, idx(i,1)] = 0.5*dx     # b_i
        A[rowV, idx(i,2)] = 1/3*dx^2   # c_i
        A[rowV, idx(i,3)] = 0.25*dx^3  # d_i
        rhs[rowV] = H[i]

        # 2) Value continuity eq : a_i + b_i Δx + c_i Δx^2 + d_i Δx^3 = a_{i+1}
        rowC1 = 4*(i-1) + 2
        A[rowC1, idx(i,0)] = 1.0      # a_i
        A[rowC1, idx(i,1)] = dx       # b_i
        A[rowC1, idx(i,2)] = dx^2     # c_i
        A[rowC1, idx(i,3)] = dx^3     # d_i
        A[rowC1, idx(ip1,0)] = -1.0   # -a_{i+1}
        rhs[rowC1] = 0.0

        # 3) Derivative continuity eq : b_i + 2 c_i Δx + 3 d_i Δx^2 = b_{i+1}
        rowC2 = 4*(i-1) + 3
        A[rowC2, idx(i,1)] = 1.0      # b_i
        A[rowC2, idx(i,2)] = 2.0*dx   # c_i
        A[rowC2, idx(i,3)] = 3.0*dx^2 # d_i
        A[rowC2, idx(ip1,1)] = -1.0   # -b_{i+1}
        rhs[rowC2] = 0.0

        # 4) Second derivative continuity eq : 2 c_i + 6 d_i Δx = 2 c_{i+1}
        rowC3 = 4*(i-1) + 4
        A[rowC3, idx(i,2)] = 2.0      # c_i
        A[rowC3, idx(i,3)] = 6.0*dx   # d_i
        A[rowC3, idx(ip1,2)] = -2.0   # -c_{i+1}
        rhs[rowC3] = 0.0
    end

    # 5) BC : a_1 = a_{n_x}, b_1 = b_{n_x}, c_1 = c_{n_x}, d_1 = d_{n_x} (periodicity)
    # Replace the last three equations for boundary conditions
    A[N-2, idx(1,0)] = 1.0
    A[N-2, idx(nx,0)] = -1.0
    rhs[N-2] = 0.0

    A[N-1, idx(1,1)] = 1.0
    A[N-1, idx(nx,1)] = -1.0
    rhs[N-1] = 0.0

    A[N, idx(1,2)] = 1.0
    A[N, idx(nx,2)] = -1.0
    rhs[N] = 0.0

    s = A \ rhs

    a, b, c, d = s[1:4:end], s[2:4:end], s[3:4:end], s[4:4:end]

    # Fonction d'interpolation
    function h_tilde(x_val)
        # Trouver dans quelle cellule se trouve x_val
        i = 1
        while i <= nx && !(x[i] <= x_val && x_val <= x[i+1])
            i += 1
        end
        
        if i > nx
            return 0.0  # En dehors du domaine
        end
        
        # Interpolation cubique entre les interfaces
        xi = x_val - x[i]
        dx_i = x[i+1] - x[i]
        return a[i] + b[i]*xi + c[i]*xi^2 + d[i]*xi^3
    end

    return h_tilde, s
end

# Utilisation
h_interp_cubic, interface_values = height_interpolation_cubic(x, H)

# Vérification de la conservation du volume
errors_cubic = verify_conservation(h_interp_cubic, x, H, 3)
println("Erreurs maximales de conservation (cubique): ", maximum(abs.(errors_cubic)))

# Tracé des résultats
h_vals_cubic = [h_interp_cubic(xi) for xi in x_fine]

fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Hauteur", title="Interpolation cubique conservant le volume")
lines!(ax, x_fine, h_vals_cubic, label="Interpolation cubique h_tilde(x)", linewidth=2)
scatter!(ax, xc, H, label="Hauteurs originales H", markersize=6)
#scatter!(ax, x, interface_values, label="Valeurs aux interfaces", markersize=4, color=:green)
axislegend(ax)
display(fig)

# Interpolation quartique
function height_interpolation_quartic(x_mesh, H)
    nx = length(H)
    N = 5*nx
    
    # Points aux interfaces
    x = x_mesh
    xc = 0.5 * (x[1:end-1] + x[2:end])
    dx = x[2] - x[1]

    # Valeurs aux interfaces (à déterminer)
    # We need 5*nx equations for 5*nx unknowns
    # We have 5 equations per cell (nx cells), but need to replace four equations for BCs
    A = zeros(N, N)
    rhs = zeros(N)

    function idx(i, locale)
        return 5*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i = a_i + 0.5 Δx b_i + 1/3 Δx^2 c_i + 0.25 Δx^3 d_i + 1/5 Δx^4 e_i
        rowV = 5*(i-1) + 1
        A[rowV, idx(i,0)] = 1.0        # a_i
        A[rowV, idx(i,1)] = 0.5*dx     # b_i
        A[rowV, idx(i,2)] = 1/3*dx^2   # c_i
        A[rowV, idx(i,3)] = 0.25*dx^3  # d_i
        A[rowV, idx(i,4)] = 1/5*dx^4   # e_i
        rhs[rowV] = H[i]

        # 2) Value continuity eq : a_i + b_i Δx + c_i Δx^2 + d_i Δx^3 + e_i Δx^4 = a_{i+1}
        rowC1 = 5*(i-1) + 2
        A[rowC1, idx(i,0)] = 1.0      # a_i
        A[rowC1, idx(i,1)] = dx       # b_i
        A[rowC1, idx(i,2)] = dx^2     # c_i
        A[rowC1, idx(i,3)] = dx^3     # d_i
        A[rowC1, idx(i,4)] = dx^4     # e_i
        A[rowC1, idx(ip1,0)] = -1.0   # -a_{i+1}
        rhs[rowC1] = 0.0

        # 3) Derivative continuity eq : b_i + 2 c_i Δx + 3 d_i Δx^2 + 4 e_i Δx^3 = b_{i+1}
        rowC2 = 5*(i-1) + 3
        A[rowC2, idx(i,1)] = 1.0      # b_i
        A[rowC2, idx(i,2)] = 2.0*dx   # c_i
        A[rowC2, idx(i,3)] = 3.0*dx^2 # d_i
        A[rowC2, idx(i,4)] = 4.0*dx^3 # e_i
        A[rowC2, idx(ip1,1)] = -1.0   # -b_{i+1}
        rhs[rowC2] = 0.0

        # 4) Second derivative continuity eq : 2 c_i + 6 d_i Δx + 12 e_i Δx^2 = 2 c_{i+1}
        rowC3 = 5*(i-1) + 4
        A[rowC3, idx(i,2)] = 2.0      # c_i
        A[rowC3, idx(i,3)] = 6.0*dx   # d_i
        A[rowC3, idx(i,4)] = 12.0*dx^2 # e_i
        A[rowC3, idx(ip1,2)] = -2.0   # -c_{i+1}
        rhs[rowC3] = 0.0

        # 5) Third derivative continuity eq : 6 d_i + 24 e_i Δx = 6 d_{i+1}
        rowC4 = 5*(i-1) + 5
        A[rowC4, idx(i,3)] = 6.0      # d_i
        A[rowC4, idx(i,4)] = 24.0*dx  # e_i
        A[rowC4, idx(ip1,3)] = -6.0   # -d_{i+1}
        rhs[rowC4] = 0.0
    end

    # 6) BC : a_1 = a_{n_x}, b_1 = b_{n_x}, c_1 = c_{n_x}, d_1 = d_{n_x}, e_1 = e_{n_x} (periodicity)
    # Replace the last four equations for boundary conditions
    A[N-3, idx(1,0)] = 1.0
    A[N-3, idx(nx,0)] = -1.0
    rhs[N-3] = 0.0

    A[N-2, idx(1,1)] = 1.0
    A[N-2, idx(nx,1)] = -1.0
    rhs[N-2] = 0.0

    A[N-1, idx(1,2)] = 1.0
    A[N-1, idx(nx,2)] = -1.0
    rhs[N-1] = 0.0

    A[N, idx(1,3)] = 1.0
    A[N, idx(nx,3)] = -1.0
    rhs[N] = 0.0

    s = A \ rhs

    a, b, c, d, e = s[1:5:end], s[2:5:end], s[3:5:end], s[4:5:end], s[5:5:end]

    # Fonction d'interpolation
    function h_tilde(x_val)
        # Trouver dans quelle cellule se trouve x_val
        i = 1
        while i <= nx && !(x[i] <= x_val && x_val <= x[i+1])
            i += 1
        end
        
        if i > nx
            return 0.0  # En dehors du domaine
        end
        
        # Interpolation quartique entre les interfaces
        xi = x_val - x[i]
        dx_i = x[i+1] - x[i]
        return return a[i] + b[i]*xi + c[i]*xi^2 + d[i]*xi^3 + e[i]*xi^4
    end

    return h_tilde, s
end

# Utilisation
h_interp_quartic, interface_values = height_interpolation_quartic(x, H)

# Vérification de la conservation du volume
errors_quartic = verify_conservation(h_interp_quartic, x, H, 4)
println("Erreurs maximales de conservation (quartique): ", maximum(abs.(errors_quartic)))

# Tracé des résultats
h_vals_quartic = [h_interp_quartic(xi) for xi in x_fine]

fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Hauteur", title="Interpolation quartique conservant le volume")
lines!(ax, x_fine, h_vals_quartic, label="Interpolation quartique h_tilde(x)", linewidth=2)
scatter!(ax, xc, H, label="Hauteurs originales H", markersize=6)
#scatter!(ax, x, interface_values, label="Valeurs aux interfaces", markersize=4, color=:green)
axislegend(ax)


# Plot the 4 interpolations
fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Hauteur", title="Interpolations linéaire et quadratique")
lines!(ax, x_fine, h_vals_linear, label="Interpolation linéaire", linewidth=2)
lines!(ax, x_fine, h_vals_quadratic, label="Interpolation quadratique", linewidth=2)
lines!(ax, x_fine, h_vals_cubic, label="Interpolation cubique", linewidth=2)
lines!(ax, x_fine, h_vals_quartic, label="Interpolation quartique", linewidth=2)
scatter!(ax, xc, H, label="Hauteurs originales H", markersize=6)
axislegend(ax)
display(fig)

# Plot on a histogram the errors : Log10 Volume Error VS 
fig = Figure()
ax = Axis(fig[1,1], xlabel="Position x", ylabel="Log10 Erreur", title="Erreurs de conservation")
scatter!(ax, xc, (abs.(errors_linear)), label="Interpolation linéaire", markersize=6)
scatter!(ax, xc, (abs.(errors_quadratic)), label="Interpolation quadratique", markersize=6)
scatter!(ax, xc, (abs.(errors_cubic)), label="Interpolation cubique", markersize=6)
scatter!(ax, xc, (abs.(errors_quartic)), label="Interpolation quartique", markersize=6)
axislegend(ax)
display(fig)

# Create a better visualization for error comparison
fig = Figure(size=(800, 500))
ax = Axis(fig[1,1], 
    xlabel="Polynomial Degree", 
    ylabel="Log10(Error)", 
    title="Volume Conservation Errors by Polynomial Degree",
    xticks=(1:4, ["Linear", "Quadratic", "Cubic", "Quartic"]))

# Data preparation
order = [1, 2, 3, 4]
errors = [errors_linear, errors_quadratic, errors_cubic, errors_quartic]
max_errors = [log10(maximum(abs.(err))) for err in errors]
avg_errors = [log10(sum(abs.(err))/length(err)) for err in errors]

# Positions for grouped bars
bar_width = 0.35
positions_max = order .- bar_width/2
positions_avg = order .+ bar_width/2

# Plot bars side by side instead of stacked
barplot!(ax, positions_max, max_errors, color=:blue, label="Maximum Error", width=bar_width)
barplot!(ax, positions_avg, avg_errors, color=:red, label="Average Error", width=bar_width)

# Add error values as text labels
for (i, val) in enumerate(max_errors)
    text!(ax, "$(round(10^val, digits=9))", 
          position=(positions_max[i], val + 0.2), 
          align=(:center, :bottom),
          fontsize=12)
end

for (i, val) in enumerate(avg_errors)
    text!(ax, "$(round(10^val, digits=9))", 
          position=(positions_avg[i], val + 0.2), 
          align=(:center, :bottom),
          fontsize=12)
end

# Add a horizontal line for machine precision
hlines!(ax, log10(eps()), linestyle=:dash, color=:black, label="Machine Precision")
text!(ax, "Machine Precision (≈2.22e-16)", 
      position=(2.5, log10(eps()) + 0.4), 
      align=(:center, :bottom),
      fontsize=12)

axislegend(ax)

display(fig)