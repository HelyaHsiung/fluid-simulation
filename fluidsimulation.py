import numpy as np
import random


class FluidSimulation:
    def __init__(self,
                 width=128,
                 height=128,
                 gasRelease=55,
                 gasLocationX=64,
                 gasLocationY=64,
                 diffusion=0.0001,
                 viscosity=0,

                 wind_location_density=4,

                 simu_windDirection=0.0,
                 simu_windSpeed=0.04,
                 simu_wind_step=50,
                 simu_windDirectionNoise_range=90,  # degree
                 simu_windSpeedNoise_range=0.3,     # percent

                 real_experiments=False,
                 real_time_points=np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900]),
                 real_windDirection=np.array([11, 12, 40, 33, 180, 22, 90, 33, 1, 270]),
                 real_windSpeed=np.array([0.01, 0.02, 0.03, 0.04, 0.03, 0.02, 0.01, 0.1, 0.001, 0.05]),
                 real_windDirectionNoise_range=60,   # degree
                 real_windSpeedNoise_range=0.2       # percent
                 ):
        self.w = width
        self.h = height
        self.FPS = 30            # Frames per second

        # Two extra cells in each dimension for the boundaries
        self.numOfCells = (height + 2) * (width + 2)

        self.i = np.arange(1, self.w+1)
        self.j = np.arange(1, self.h+1)
        ii, jj = np.meshgrid(self.i, self.j)
        self.ii = ii.ravel()
        self.jj = jj.ravel()

        self.Iij = self.I(self.ii, self.jj)

        self.Iij0 = self.I(self.ii - 1, self.jj)
        self.Iij1 = self.I(self.ii + 1, self.jj)
        self.Iij2 = self.I(self.ii, self.jj - 1)
        self.Iij3 = self.I(self.ii, self.jj + 1)

        self.gasRelease = gasRelease
        self.gasLocationX = int(gasLocationX)
        self.gasLocationY = int(gasLocationY)
        self.diffusion = diffusion # The amount of diffusion
        self.viscosity = viscosity # The fluid's viscosity

        self.dt = 0.1 # The simulation time-step

        # Number of iterations to use in the Gauss-Seidel method in linearSolve()
        self.iterations = 20

        # Add details
        self.doVorticityConfinement = True
        self.doBuoyancy = False

        # Location of wind
        wind_ii, wind_jj = np.meshgrid(np.arange(0, self.w, wind_location_density),
                                       np.arange(0, self.h, wind_location_density))
        self.wind_Locations = self.I(wind_ii.ravel(), wind_jj.ravel())

        # Change wind direction after N-th timesteps
        self.simu_windDirection = simu_windDirection
        self.simu_windSpeed = simu_windSpeed
        self.simu_wind_step = simu_wind_step
        self.simu_windDirectionNoise_range = simu_windDirectionNoise_range
        self.simu_windSpeedNoise_range = simu_windSpeedNoise_range
        self.simu_windDirectionNoise_cur = 0
        self.simu_windSpeedNoise_cur = 0
        # Real wind noise
        self.real_experiments = real_experiments
        self.real_time_points = real_time_points
        self.real_windDirection = real_windDirection
        self.real_windSpeed = real_windSpeed
        self.real_windDirectionNoise_range = real_windDirectionNoise_range
        self.real_windSpeedNoise_range = real_windSpeedNoise_range

        # Values for current simulation step, and initialize everything to zero
        self.u = np.zeros(self.numOfCells) # Velocity x
        self.v = np.zeros(self.numOfCells) # Velocity y
        self.d = np.zeros(self.numOfCells) # Density

        # Values from the last simulation step
        self.uOld = np.zeros(self.numOfCells)
        self.vOld = np.zeros(self.numOfCells)
        self.dOld = np.zeros(self.numOfCells)

        self.curlData = np.zeros(self.numOfCells) # The cell's curl

        # Boundaries enumeration
        self.BOUNDARY_NONE = 0
        self.BOUNDARY_LEFT_RIGHT = 1
        self.BOUNDARY_TOP_BOTTOM = 2

        # counter
        self.index_step = 0

    """
    Index
    """
    def I(self, i, j):
        return i + (self.w + 2) * j

    """
    Density step.
    """
    def densityStep(self):
        self.addSource(self.d, self.dOld)

        self.swapD()
        self.diffuse(self.BOUNDARY_NONE, self.d, self.dOld, self.diffusion)

        self.swapD()
        self.advect(self.BOUNDARY_NONE, self.d, self.dOld, self.u, self.v)

        # Reset for next step
        self.dOld.fill(0)

    """
    Velocity step.
    """
    def velocityStep(self):
        self.addSource(self.u, self.uOld)
        self.addSource(self.v, self.vOld)

        if (self.doVorticityConfinement):
            self.vorticityConfinement(self.uOld, self.vOld)
            self.addSource(self.u, self.uOld)
            self.addSource(self.v, self.vOld)

        if (self.doBuoyancy):
            self.buoyancy(self.vOld)
            self.addSource(self.v, self.vOld)
        self.swapU()
        self.diffuse(self.BOUNDARY_LEFT_RIGHT, self.u, self.uOld, self.viscosity)

        self.swapV()
        self.diffuse(self.BOUNDARY_TOP_BOTTOM, self.v, self.vOld, self.viscosity)

        self.project(self.u, self.v, self.uOld, self.vOld)
        self.swapU()
        self.swapV()

        self.advect(self.BOUNDARY_LEFT_RIGHT, self.u, self.uOld, self.uOld, self.vOld)
        self.advect(self.BOUNDARY_TOP_BOTTOM, self.v, self.vOld, self.uOld, self.vOld)

        self.project(self.u, self.v, self.uOld, self.vOld)

        # Reset for next step
        self.uOld.fill(0)
        self.vOld.fill(0)

    """
    Resets the density.
    """
    def resetDensity(self):
        self.d.fill(0)

    """
    Resets the velocity.
    """
    def resetVelocity(self):
        self.v.fill(0.001)
        self.u.fill(0.001)

    """
    Swap velocity x reference.
    """
    def swapU(self):
        self.u, self.uOld = self.uOld, self.u

    """
    Swap velocity y reference.
    """
    def swapV(self):
        self.v, self.vOld = self.vOld, self.v

    """
    Swap density reference.
    """
    def swapD(self):
        self.d, self.dOld = self.dOld, self.d

    """
    Integrate the density sources.
    """
    def addSource(self, x, s):
        x += s * self.dt

    """
    Calculate the curl at cell (i, j)
    This represents the vortex strength at the cell.
    Computed as: w = (del x U) where U is the velocity vector at (i, j).
    """
    def curl(self, i, j):
        duDy = (self.u[self.I(i, j + 1)] - self.u[self.I(i, j - 1)]) * 0.5
        dvDx = (self.v[self.I(i + 1, j)] - self.v[self.I(i - 1, j)]) * 0.5

        return duDy - dvDx

    """
    Calculate the vorticity confinement force for each cell.
    Fvc = (N x W) where W is the curl at (i, j) and N = del |W| / |del |W||.
    N is the vector pointing to the vortex center, hence we
    add force perpendicular to N.
    """
    def vorticityConfinement(self, vcX, vcY):
        # Calculate magnitude of curl(i, j) for each cell
        self.curlData[self.Iij] = np.abs(self.curl(self.ii, self.jj))

        dx = (self.curlData[self.Iij1] - self.curlData[self.Iij0]) * 0.5
        dy = (self.curlData[self.Iij3] - self.curlData[self.Iij2]) * 0.5
        
        norm = np.sqrt(dx*dx + dy*dy)
        norm[norm == 0] = 1
        
        dx /= norm
        dy /= norm
        
        v = self.curl(self.ii, self.jj)
        
        vcX[self.Iij] = dy * v * -1
        vcY[self.Iij] = dx * v

    """
    Calculate the buoyancy force for the grid.
    Fbuoy = -a * d * Y + b * (T - Tamb) * Y where Y = (0,1)
    The constants a and b are positive with physically meaningful quantities.
    T is the temperature at the current cell, Tamb is the average temperature of the fluid grid
    
    In this simplified implementation we say that the temperature is synonymous with density
    and because there are no other heat sources we can just use the density field instead of adding a new
    temperature field.
    """
    def buoyancy(self, buoy):
        a = 0.000625
        b = 0.025
        
        # Calculate average temperature of the grid
        tAmb = np.sum(self.d) / (self.w * self.h)

        # For each cell compute buoyancy force
        buoy[self.Iij] = a * self.d[self.Iij] - b * (self.d[self.Iij] - tAmb)


    """
    Diffuse the density between neighbouring cells.
    """
    def diffuse(self, b, x, x0, diffusion):
        a = self.dt * diffusion * self.w * self.h

        self.linearSolve(b, x, x0, a, 1 + 4 * a)

    """
    The advection step moves the density through the static velocity field.
    Instead of moving the cells forward in time, we treat the cell's center as a particle
    and then trace it back in time to look for the 'particles' which end up at the cell's center.
    """
    def advect(self, b, d, d0, u, v):
        x = self.ii - self.dt * self.w * u[self.Iij]
        y = self.jj - self.dt * self.h * v[self.Iij]
        
        x[x < 0.5] = 0.5
        x[x > self.w + 0.5] = self.w + 0.5
        
        i0 = x.astype(int)
        i1 = i0 + 1
        
        y[y < 0.5] = 0.5
        y[y > self.h + 0.5] = self.h + 0.5
        
        j0 = y.astype(int)
        j1 = j0 + 1
        
        s1 = x - i0
        s0 = 1 - s1
        
        t1 = y - j0
        t0 = 1 - t1

        d[self.Iij] = s0 * (t0 * d0[self.I(i0, j0)] + t1 * d0[self.I(i0, j1)]) + s1 * (t0 * d0[self.I(i1, j0)] + t1 * d0[self.I(i1, j1)])

        self.setBoundary(b, d)

    """
    Forces the velocity field to be mass conserving.
    This step is what actually produces the nice looking swirly vortices.
    
    It uses a result called Hodge Decomposition which says that every velocity field is the sum
    of a mass conserving field, and a gradient field. So we calculate the gradient field, and subtract
    it from the velocity field to get a mass conserving one.
    It solves a linear system of equations called Poisson Equation.
    """
    def project(self, u, v, p, div):
        # Calculate the gradient field
        div[self.Iij] = -0.5 * ((u[self.Iij1] - u[self.Iij0])/self.w + (v[self.Iij3] - v[self.Iij2])/self.h)
        
        p.fill(0.0)

        self.setBoundary(self.BOUNDARY_NONE, div)
        self.setBoundary(self.BOUNDARY_NONE, p)

        # Solve the Poisson equations
        self.linearSolve(self.BOUNDARY_NONE, p, div, 1, 4)

        # Subtract the gradient field from the velocity field to get a mass conserving velocity field.
        u[self.Iij] -= 0.5 * (p[self.Iij1] - p[self.Iij0]) * self.w
        v[self.Iij] -= 0.5 * (p[self.Iij3] - p[self.Iij2]) * self.h

        self.setBoundary(self.BOUNDARY_LEFT_RIGHT, u)
        self.setBoundary(self.BOUNDARY_TOP_BOTTOM, v)

    """
    Solve a linear system of equations using Gauss-Seidel method.
    """
    def linearSolve(self, b, x, x0, a, c):
        invC = 1.0 / c
        for k in range(self.iterations):
            x[self.Iij] = (x0[self.Iij] + a*(x[self.Iij0] + x[self.Iij1] + x[self.Iij2] + x[self.Iij3])) * invC
            self.setBoundary(b, x)

    """
    Set boundary conditions.
    """
    def setBoundary(self, b, x):
        if (b == self.BOUNDARY_LEFT_RIGHT):
            x[self.I(0, self.j)] = -x[self.I(1, self.j)]
            x[self.I(self.w + 1, self.j)] = -x[self.I(self.w, self.j)]
        else:
            x[self.I(0, self.j)].fill(0.0)
            x[self.I(self.w + 1, self.j)].fill(0.0)

        if (b == self.BOUNDARY_TOP_BOTTOM):
            x[self.I(self.i, 0)] = -x[self.I(self.i, 1)]
            x[self.I(self.i, self.h + 1)] = -x[self.I(self.i, self.h)]
        else:
            x[self.I(self.i, 0)].fill(0.0)
            x[self.I(self.i, self.h + 1)].fill(0.0)

        x[self.I(0, 0)] = 0.5 * (x[self.I(1, 0)] + x[self.I(0, 1)])
        x[self.I(0, self.h + 1)] = 0.5 * (x[self.I(1, self.h + 1)] + x[self.I(0, self.h)])
        x[self.I(self.w + 1, 0)] = 0.5 * (x[self.I(self.w, 0)] + x[self.I(self.w + 1, 1)])
        x[self.I(self.w + 1, self.h + 1)] = 0.5 * (x[self.I(self.w, self.h + 1)] + x[self.I(self.w + 1, self.h)])

    def toRadians (self, angle):
        return angle * (np.pi/ 180)

    def update(self, f):
        self.dOld[int(self.I(self.gasLocationX, self.gasLocationY))] = self.gasRelease
        # Step the fluid simulation
        self.velocityStep()
        self.densityStep()
        if self.real_experiments:
            latest_ealier = np.sum(self.real_time_points<=self.index_step) - 1
            windDirection = self.real_windDirection[latest_ealier]
            windSpeed = self.real_windSpeed[latest_ealier]
            windDirection += (random.random()*2 - 1) * self.real_windDirectionNoise_range
            windSpeed *= (1 + (random.random()*2 - 1) * self.real_windSpeedNoise_range)
        else:
            if self.index_step % self.simu_wind_step == 0:
                self.simu_windDirectionNoise_cur = (random.random()*2 - 1) * self.simu_windDirectionNoise_range
                self.simu_windSpeedNoise_cur = (random.random()*2 - 1) * self.simu_windSpeedNoise_range
            else:
                pass
            windDirection = self.simu_windDirection + self.simu_windDirectionNoise_cur
            windSpeed = self.simu_windSpeed * (1 + self.simu_windSpeedNoise_cur)

        self.uOld[self.wind_Locations] = np.cos(self.toRadians(windDirection)) * windSpeed
        self.vOld[self.wind_Locations] = np.sin(self.toRadians(windDirection)) * windSpeed

        dset = f.create_dataset(f"frame{self.index_step:03d}", (self.h, self.w), dtype='f', compression="gzip")
        density_map = self.d.reshape((self.h + 2, self.w + 2))[1:self.h + 1, 1:self.w + 1]
        dset[...] = self.gasRelease * ((density_map - density_map.min()) / (density_map.max() - density_map.min()))
        self.index_step += 1
