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

Particle sedimentation velocity:

$$V_0 = \frac{g (\rho_p - \rho_f) d^2}{18 \nu \rho_f}$$

==========================================================================
=== Швидкість осідання (формула Стокса) ===
==========================================================================
  № | Material   |     ρp |    d [m] |    ρf |   ν [m²/s] | V_settle [m/s]
--------------------------------------------------------------------------
  0 | Abrasive   |   2400 |  5.0e-05 |  1000 |    1.0e-06 |     1.9075e-03
  1 | Abrasive   |   2400 |  5.0e-05 |  1000 |    1.0e-05 |     1.9075e-04
  2 | Abrasive   |   2400 |  5.0e-05 |  1000 |    1.0e-04 |     1.9075e-05
  3 | Abrasive   |   2400 |  1.5e-04 |  1000 |    1.0e-06 |     1.7167e-02
  4 | Abrasive   |   2400 |  1.5e-04 |  1000 |    1.0e-05 |     1.7167e-03
  5 | Abrasive   |   2400 |  1.5e-04 |  1000 |    1.0e-04 |     1.7167e-04
  6 | Abrasive   |   2400 |  2.5e-04 |  1000 |    1.0e-06 |     4.7687e-02
  7 | Abrasive   |   2400 |  2.5e-04 |  1000 |    1.0e-05 |     4.7687e-03
  8 | Abrasive   |   2400 |  2.5e-04 |  1000 |    1.0e-04 |     4.7687e-04
  9 | Abrasive   |   2400 |  5.0e-04 |  1000 |    1.0e-06 |     1.9075e-01
 10 | Abrasive   |   2400 |  5.0e-04 |  1000 |    1.0e-05 |     1.9075e-02
 11 | Abrasive   |   2400 |  5.0e-04 |  1000 |    1.0e-04 |     1.9075e-03
 12 | Metallic   |   7800 |  5.0e-05 |  1000 |    1.0e-06 |     9.2650e-03
 13 | Metallic   |   7800 |  5.0e-05 |  1000 |    1.0e-05 |     9.2650e-04
 14 | Metallic   |   7800 |  5.0e-05 |  1000 |    1.0e-04 |     9.2650e-05
 15 | Metallic   |   7800 |  1.5e-04 |  1000 |    1.0e-06 |     8.3385e-02
 16 | Metallic   |   7800 |  1.5e-04 |  1000 |    1.0e-05 |     8.3385e-03
 17 | Metallic   |   7800 |  1.5e-04 |  1000 |    1.0e-04 |     8.3385e-04
 18 | Metallic   |   7800 |  2.5e-04 |  1000 |    1.0e-06 |     2.3162e-01
 19 | Metallic   |   7800 |  2.5e-04 |  1000 |    1.0e-05 |     2.3162e-02
 20 | Metallic   |   7800 |  2.5e-04 |  1000 |    1.0e-04 |     2.3162e-03
 21 | Metallic   |   7800 |  5.0e-04 |  1000 |    1.0e-06 |     9.2650e-01
 22 | Metallic   |   7800 |  5.0e-04 |  1000 |    1.0e-05 |     9.2650e-02
 23 | Metallic   |   7800 |  5.0e-04 |  1000 |    1.0e-04 |     9.2650e-03

================================================================================
=== Швидкість входу частинок у рідину (без тертя) ===
================================================================================
  № |     L |     h |    θ |  V_chute |       Vx |       Vy |  |V_imp| |  Angle°
--------------------------------------------------------------------------------
  0 |   0.5 |  0.00 |   30 |   2.2147 |   1.9180 |   1.1074 |   2.2147 |   30.00
  1 |   0.5 |  0.00 |   45 |   2.6338 |   1.8624 |   1.8624 |   2.6338 |   45.00
  2 |   0.5 |  0.05 |   30 |   2.2147 |   1.9180 |   1.4857 |   2.4261 |   37.76
  3 |   0.5 |  0.05 |   45 |   2.6338 |   1.8624 |   2.1094 |   2.8138 |   48.56
  4 |   0.5 |  0.10 |   30 |   2.2147 |   1.9180 |   1.7856 |   2.6205 |   42.95
  5 |   0.5 |  0.10 |   45 |   2.6338 |   1.8624 |   2.3303 |   2.9831 |   51.37
  6 |   1.0 |  0.00 |   30 |   3.1321 |   2.7125 |   1.5660 |   3.1321 |   30.00
  7 |   1.0 |  0.00 |   45 |   3.7247 |   2.6338 |   2.6338 |   3.7247 |   45.00
  8 |   1.0 |  0.05 |   30 |   3.1321 |   2.7125 |   1.8530 |   3.2850 |   34.34
  9 |   1.0 |  0.05 |   45 |   3.7247 |   2.6338 |   2.8138 |   3.8541 |   46.89
 10 |   1.0 |  0.10 |   30 |   3.1321 |   2.7125 |   2.1011 |   3.4310 |   37.76
 11 |   1.0 |  0.10 |   45 |   3.7247 |   2.6338 |   2.9831 |   3.9794 |   48.56
 12 |   1.5 |  0.00 |   30 |   3.8360 |   3.3221 |   1.9180 |   3.8360 |   30.00
 13 |   1.5 |  0.00 |   45 |   4.5618 |   3.2257 |   3.2257 |   4.5618 |   45.00
 14 |   1.5 |  0.05 |   30 |   3.8360 |   3.3221 |   2.1586 |   3.9618 |   33.02
 15 |   1.5 |  0.05 |   45 |   4.5618 |   3.2257 |   3.3743 |   4.6681 |   46.29
 16 |   1.5 |  0.10 |   30 |   3.8360 |   3.3221 |   2.3750 |   4.0837 |   35.56
 17 |   1.5 |  0.10 |   45 |   4.5618 |   3.2257 |   3.5167 |   4.7720 |   47.47

===============================================
=== Об'ємна концентрація α і масова витрата ===
===============================================
Площа входу A_inlet = 0.0032 м²
Швидкість входу v_imp = 2.426 м/с
Густина частинок ρp = 7800.0 кг/м³

  m [кг/с] |   V̇_p [м³/с] |          α
--------------------------------------
       0.2 |     0.000026 |    0.00330
       0.5 |     0.000064 |    0.00826
       0.6 |     0.000077 |    0.00991
       1.0 |     0.000128 |    0.01651
       2.5 |     0.000321 |    0.04129
       5.0 |     0.000641 |    0.08257
      10.0 |     0.001282 |    0.16514

         α |   V̇_p [м³/с] |   m [кг/с]
--------------------------------------
     0.010 |     0.000078 |      0.606
     0.025 |     0.000194 |      1.514
     0.050 |     0.000388 |      3.028
     0.075 |     0.000582 |      4.541
     0.100 |     0.000776 |      6.055
     0.250 |     0.001941 |     15.138
     0.500 |     0.003882 |     30.276



<video src="https://github.com/user-attachments/assets/dbdb00b0-aec7-4264-8d7d-d6036002e270" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/49e5469b-eeba-4c27-91ac-3b55c52579af" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/39f493d7-1773-4454-8a1c-a536851a3bea" autoplay loop muted playsinline width="100%"></video>

<video src="https://github.com/user-attachments/assets/dd856e40-b103-413b-8d9b-1a727b68687f" autoplay loop muted playsinline width="100%"></video>

Розрахунок 7

[![Розрахунок 7 (концентрація твердої фази)](https://img.youtube.com/vi/ot0rxoV9tZM/maxresdefault.jpg)](https://youtu.be/ot0rxoV9tZM)

[![Розрахунок 7 (швидкості)](https://img.youtube.com/vi/bBP0aTuADU4/maxresdefault.jpg)](https://youtu.be/bBP0aTuADU4)

Розрахунок 13

[![Розрахунок 13 (концентрація твердої фази)](https://img.youtube.com/vi/30mSTdmtIP4/maxresdefault.jpg)](https://youtu.be/30mSTdmtIP4)

[![Розрахунок 13 (швидкості)](https://img.youtube.com/vi/NXW3bXX27i8/maxresdefault.jpg)](https://youtu.be/NXW3bXX27i8)

Розрахунок 16

[![Розрахунок 16 (концентрація твердої фази)](https://img.youtube.com/vi/kSMuOxZtJaA/maxresdefault.jpg)](https://youtu.be/kSMuOxZtJaA)

[![Розрахунок 16 (швидкості)](https://img.youtube.com/vi/p5D3x6vhbiE/maxresdefault.jpg)](https://youtu.be/p5D3x6vhbiE)

Розрахунок 17

[![Розрахунок 17 (концентрація твердої фази)](https://img.youtube.com/vi/OwlW1psPOeE/maxresdefault.jpg)](https://youtu.be/OwlW1psPOeE)

[![Розрахунок 17 (швидкості)](https://img.youtube.com/vi/2VbWSX5bDoU/maxresdefault.jpg)](https://youtu.be/2VbWSX5bDoU)

Розрахунок 18

[![Розрахунок 18 (концентрація твердої фази)](https://img.youtube.com/vi/3ML1kLi12VI/maxresdefault.jpg)](https://youtu.be/3ML1kLi12VI)

[![Розрахунок 18 (швидкості)](https://img.youtube.com/vi/LPAqQJyPVhs/maxresdefault.jpg)](https://youtu.be/LPAqQJyPVhs)

Зі змінененою швидкістю входу твердої фази

[![particle](https://img.youtube.com/vi/vSyLmfuDLD4/maxresdefault.jpg)](https://youtu.be/vSyLmfuDLD4)

