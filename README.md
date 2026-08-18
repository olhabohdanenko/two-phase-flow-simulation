# Mathematical Model of Solid-Liquid Medium Flow

This paper presents a mathematical model of the solid-liquid medium (suspension) flow. To model the movement of solid particles in a fluid stream, we state the following assumptions:

1. Interaction between particles is neglected, as their sizes and masses are insignificant compared to the fluid flow;
2. The impact of particles on the surrounding medium (fluid) is minimal, since the number of particles is much smaller than the amount of fluid in the system;
3. The size and mass of the particles are considered constant, as well as the density of the fluid.

That is, we have two continuous media where the fluid transports the particles. The number of these particles is small, and they do not dissolve in the fluid. The motion of the medium can be represented by the following system of equations:

$$
\begin{aligned}
\frac{d \vec{v}}{dt} &= - \vec{\nabla} p' + \nu_{1} \left( 1 + \beta \left( A + y \right) \right) \nabla^{2} \vec{v} + \\
&\quad + \left( \zeta' + \frac{\nu_{1}}{3} +\beta \left( \zeta' y + \frac{\nu_{1}}{3} \left( A + y \right) \right) \right)  \vec{\nabla} \left( \vec{\nabla} \cdot \vec{v} \right) - \\
&\quad - \beta y \vec{g} - x \nabla^{k} \left( \beta \vec{w}_{2} w_{2}^{k} \right)
\end{aligned}
$$

$$
\vec{\nabla} \cdot \vec{v} = y \left[ - \vec{\nabla} \cdot \left( \beta \vec{w}_{2} \right) + \phi \right] 
$$

where $\vec{v}$ is the barycentric velocity of the medium as a whole, $p' = \frac{p_{d}}{\rho_{1}^{0}}$ is the dynamic part of the pressure normalized to the fluid density, $\nu_{1} = \frac{\mu_{1}}{\rho_{1}^{0}}$ is the kinematic viscosity of the fluid, $\zeta' = \frac{\zeta}{\rho_{1}^{0}}$ is the second kinematic viscosity of the fluid, $\phi = \frac{J_{12}}{\rho_{2}^{0}}$ is the volumetric source of the solid phase, $y = 1 - x$, $x = \frac{\rho_{2}^{0}}{\rho_{1}^{0}}$, $A$ is Einstein's viscosity coefficient, $\beta$ is the volume fraction of the solid phase, $\vec{g}$ is the gravitational acceleration vector, $\vec{w}_{2}$ is the diffusion velocity of the solid phase.

$$
\vec{v}_{1} = \frac{\vec{v} - \beta x \vec{v}_{2}}{1 - \beta x}, \qquad \vec{w}_{1} = - \beta x \vec{v}_{21}, \qquad \rho_{1} =  \left( 1 - \beta \right) \rho_{1}^{0}
$$

$$
\vec{v}_{2} = \frac{\vec{v} - \left( 1 - \beta x \right) \vec{v}_{1}}{\beta x}, \qquad \vec{w}_{2} = \left( 1 - \beta x \right) \vec{v}_{21}, \qquad  \rho_{2} = \rho_{2}^{0} \beta
$$

$$
\vec{v} = \vec{v_{1}} + \beta x \vec{v}_{21}, \qquad \vec{v} = \left( 1 - \beta x \right) \vec{v_{1}} + \beta x \vec{v}_{2}, \qquad  \rho = \rho_{1}^{0} \left( 1 - \beta y \right)
$$

$$
\vec{v}_{21} = \vec{v}_{2} - \vec{v}_{1}, \qquad \vec{v}_{21} = \frac{\vec{v}_{2} - \vec{v}}{1 - \beta x}, \qquad \vec{v}_{21} = \frac{\vec{v} - \vec{v}_{1}}{\beta x}
$$

Taking into account assumption 3, the component $\phi = 0$. And considering assumption 2, the component $- x \nabla^{k} \left( \beta \vec{w}_{2} w_{2}^{k} \right)$ is neglected.

Then, we can write the velocity equation of the solid phase:

$$
\left( C_{A} + x \right) \frac{d_{2} \vec{v}_{2}}{dt} = -y \vec{g} + \vec{f}_{D} - \frac{1}{\beta} x \phi \vec{v}_2
$$

$$
\vec{f}_{D} = - \left( C_{1D} \nu_{1} \frac{\vec{v}_{21}}{d^{2}} + C_{2D} \frac{\vec{v}_{21}^{2}}{d} \right) \vec{n}_{21}
$$

where $\vec{f}_{D}$ is the specific drag force, $\vec{v}_{2}$ is the velocity of the solid phase, $\vec{v}_{21} = \vec{v}_{2} - \vec{v}_{1}$ is the relative velocity of the phases, $C_{A}$ is the added mass coefficient, $C_{1D}$ is the linear (viscous) drag coefficient (corresponding to Stokes' law), $C_{2D}$ is the quadratic (inertial) drag coefficient (corresponding to Newton's law at high Reynolds numbers). For our assumptions, we have: $C_{1D} = 18$, $C_{2D} = 0$.

The assumption for the mixture momentum equation includes the constancy of the number of particles in the selected unit volume, which is fulfilled during their coordinated collective motion; if this is not the case, an effective diffusion component is added to the equation:

$$
\vec{w}_{2}^{d} = - \frac{1}{\beta} D \vec{\nabla} \beta
$$

Then:

$$
\vec{v}_{2} = \vec{v}_{2}^{c} + \vec{w}_{2}^{d}
$$

where $\vec{v}_{2}^{c}$ is the collective component of the solid phase velocity, which satisfies the solid phase velocity equation.
With this in mind, the mass transport equation for the solid phase will take the form:

$$
\frac{\partial \beta}{\partial t} + \vec{\nabla} \cdot \left( \beta \vec{v}_{2}^{c} \right) = \vec{\nabla} \cdot \left( D \vec{\nabla} \beta \right) + \phi
$$

where $D$ is the effective coefficient describing the irregular movement of particles due to turbulent eddies.

The turbulent nature of the motion is taken into account by a two-parameter turbulence model, which is represented by an algebraic expression:

$$
\nu_{i', j}^x = \nu + \frac{\delta x}{Re} \left| u_{i', j} \right|, \quad \nu_{i', j}^z = \nu + \frac{\delta z}{Re} \left| w_{i', j} \right|
$$

$$
\nu_{i', j'}^x = \nu + \frac{1}{2}  \frac{\delta z}{Re} \left| w_{i, j'} + w_{i, j' - 1} \right|, \quad \nu_{i', j'}^z = \nu + \frac{1}{2}  \frac{\delta x}{Re} \left| u_{i', j} + u_{i' - 1, j'} \right|
$$

$$
D_{i', j}^x = D + \frac{\delta x}{Re} \left| u_{i', j} \right|, \quad D_{i', j}^z = D + \frac{\delta z}{Re} \left| w_{i', j} \right|
$$

To solve the model equations, we use the Newton-Raphson method. For this, it is necessary to build the Jacobian matrix and the right-hand side vector. It will take the form:

$$
\begin{aligned}
F_{u_{i',j}} &= \frac{u_{i',j}^{n+1} - u_{i',j}^{n}}{\tau} + \frac{1}{\rho}\frac{p_{i+1,j}^{n+1} - p_{i,j}^{n+1}}{\delta x} \\
&\quad + \left( \frac{\partial (u^{2})^{n+1}}{\partial x} + \frac{\partial (uw)^{n+1}}{\partial z} \right) \\
&\quad - \frac{\partial}{\partial x}\left[ \nu_{1}\bigl(1 + \beta^{n+1}(A+y)\bigr) \frac{\partial u^{n+1}}{\partial x} \right] \\
&\quad - \frac{\partial}{\partial z}\left[ \nu_{1}\bigl(1 + \beta^{n+1}(A+y)\bigr) \frac{\partial u^{n+1}}{\partial z} \right] \\
&\quad - \frac{\partial}{\partial x}\left[ \lambda^{n+1} \left( \frac{\partial u^{n+1}}{\partial x} + \frac{\partial w^{n+1}}{\partial z} \right) \right] = 0
\end{aligned}
$$

$$
\lambda^{n+1} = \zeta' + \frac{\nu_{1}}{3} + \beta^{n+1}\left( \zeta' y + \frac{\nu_{1}}{3}(A + y) \right)
$$

$$
\left( \vec{v} \cdot \vec{\nabla} \right) \vec{v} = \left[ \frac{\partial u^{2}}{\partial x} + \frac{\partial \left( uw \right)}{\partial z}\right]
$$

**Central differences:**

$$
\frac{\partial u^{2}}{\partial x} = \frac{u_{i' + 1, j}^{2} - u_{i' - 1, j}^{2}}{2 \delta x}
$$

$$
\frac{\partial \left( uw \right)}{\partial z} = \frac{\left( uw \right)_{i', j'} - \left( uw \right)_{i', j' - 1}}{\delta z}
$$

where

$$
\left( uw \right)_{i', j'} = \frac{1}{4} \left( u_{i', j} + u_{i', j + 1} \right) \left( w_{i, j'} + w_{i + 1, j'} \right)
$$

**Upwind:**

$$
\frac{\partial u^{2}}{\partial x} : \quad u_{i', j} > 0 : \frac{u_{i', j}^{2} - u_{i' - 1, j}^{2}}{\delta x}, \quad u_{i', j} < 0 : \frac{u_{i' + 1, j}^{2} - u_{i', j}^{2}}{\delta x}
$$

$$
\begin{aligned}
\frac{\partial \left( uw \right)}{\partial z} : \quad & w_{i', j'} > 0 : \frac{w_{i', j'} u_{i', j} - w_{i', j' - 1} u_{i', j - 1} }{\delta z}, \\
& w_{i', j'} < 0 : \frac{w_{i', j' + 1} u_{i', j + 1} - w_{i', j'} u_{i', j} }{\delta z}
\end{aligned}
$$

where

$$
w_{i', j'} = \frac{1}{2} \left( w_{i, j'} + w_{i + 1, j'} \right)
$$

**TVD:**

$$
\frac{\partial u^{2}}{\partial x} = \frac{F_{i', j} - F_{i' - 1, j}}{\delta x}
$$

$$
\frac{\partial \left( uw \right)}{\partial z} = \frac{G_{i', j'} - G_{i', j' - 1}}{\delta z}
$$

where

$$
F_{i', j} = F_{i', j}^{LO} + \phi \left( r \right) \left( F_{i', j}^{HO} - F_{i', j}^{LO}\right)
$$

$$
G_{i', j'} = G_{i', j'}^{LO} + \phi \left( r \right) \left( G_{i', j'}^{HO} - G_{i', j'}^{LO}\right)
$$

$$
F_{i', j}^{LO}: \quad u_{i', j} > 0 : u_{i', j}^{2}, \quad u_{i', j} < 0 : u_{i' + 1, j}^{2}
$$

$$
G_{i', j'}^{LO}: \quad w_{i', j'} > 0 : w_{i', j'} u_{i', j}, \quad w_{i', j'} < 0 : w_{i', j'}u_{i', j + 1}
$$

$$
F_{i', j}^{HO}: \quad \frac{1}{2} \left( u_{i', j}^{2} + u_{i' + 1, j}^{2} \right)
$$

$$
G_{i', j'}^{HO}: \quad w_{i', j'} \frac{1}{2} \left( u_{i', j} + u_{i', j + 1} \right)
$$

**For beta:**

**Central differences:**

$$
\frac{\partial \left( u_{2} \beta \right)}{\partial x} = \frac{u_{2_{i',j}} \beta_{e} - u_{2_{i' - 1,j}} \beta_{w}}{\delta x}
$$

$$
\frac{\partial \left( w_{2} \beta \right)}{\partial z} = \frac{w_{2_{i,j'}} \beta_{n} - w_{2_{i,j' - 1}} \beta_{s}}{\delta z}
$$

**Upwind:**

$$
\frac{\partial u^{2}}{\partial x} : \quad u_{i', j} > 0 : \frac{u_{i', j}^{2} - u_{i' - 1, j}^{2}}{\delta x}, \quad u_{i', j} < 0 : \frac{u_{i' + 1, j}^{2} - u_{i', j}^{2}}{\delta x}
$$

$$
\begin{aligned}
\frac{\partial \left( uw \right)}{\partial z} : \quad & w_{i', j'} > 0 : \frac{w_{i', j'} u_{i', j} - w_{i', j' - 1} u_{i', j - 1} }{\delta z}, \\
& w_{i', j'} < 0 : \frac{w_{i', j' + 1} u_{i', j + 1} - w_{i', j'} u_{i', j} }{\delta z}
\end{aligned}
$$

where

$$
w_{i', j'} = \frac{1}{2} \left( w_{i, j'} + w_{i + 1, j'} \right)
$$

**TVD:**

$$
\frac{\partial u^{2}}{\partial x} = \frac{F_{i', j} - F_{i' - 1, j}}{\delta x}
$$

$$
\frac{\partial \left( uw \right)}{\partial z} = \frac{G_{i', j'} - G_{i', j' - 1}}{\delta z}
$$

where

$$
F_{i', j} = F_{i', j}^{LO} + \phi \left( r \right) \left( F_{i', j}^{HO} - F_{i', j}^{LO}\right)
$$

$$
G_{i', j'} = G_{i', j'}^{LO} + \phi \left( r \right) \left( G_{i', j'}^{HO} - G_{i', j'}^{LO}\right)
$$

$$
F_{i', j}^{LO}: \quad u_{i', j} > 0 : u_{i', j}^{2}, \quad u_{i', j} < 0 : u_{i' + 1, j}^{2}
$$

$$
G_{i', j'}^{LO}: \quad w_{i', j'} > 0 : w_{i', j'} u_{i', j}, \quad w_{i', j'} < 0 : w_{i', j'}u_{i', j + 1}
$$

$$
F_{i', j}^{HO}: \quad \frac{1}{2} \left( u_{i', j}^{2} + u_{i' + 1, j}^{2} \right)
$$

$$
G_{i', j'}^{HO}: \quad w_{i', j'} \frac{1}{2} \left( u_{i', j} + u_{i', j + 1} \right)
$$
