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

<video src="https://github.com/user-attachments/assets/dbdb00b0-aec7-4264-8d7d-d6036002e270" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/49e5469b-eeba-4c27-91ac-3b55c52579af" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/39f493d7-1773-4454-8a1c-a536851a3bea" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/dd856e40-b103-413b-8d9b-1a727b68687f" autoplay loop muted playsinline width="100%"></video>
