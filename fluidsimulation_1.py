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

        self.gasRelease = gasRelease
        self.gasLocationX = int(gasLocationX)
        self.gasLocationY = int(gasLocationY)
        self.diffusion = diffusion # The amount of diffusion
        self.viscosity = viscosity # The fluid's viscosity

        self.dt = 0.1 # The simulation time-step

        # Number of iterations to use in the Gauss-Seidel method in linearSolve()
        self.iterations = 10

        # Add details
        self.doVorticityConfinement = True
        self.doBuoyancy = False

        # Location of wind
        self.acc = wind_location_density

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
        self.u = np.zeros((self.h+2, self.w+2)) # Velocity x
        self.v = np.zeros((self.h+2, self.w+2)) # Velocity y
        self.d = np.zeros((self.h+2, self.w+2)) # Density

        # Values from the last simulation step
        self.uOld = np.zeros((self.h+2, self.w+2))
        self.vOld = np.zeros((self.h+2, self.w+2))
        self.dOld = np.zeros((self.h+2, self.w+2))

        self.curlData = np.zeros((self.h+2, self.w+2)) # The cell's curl

        # Boundaries enumeration
        self.BOUNDARY_NONE = 0
        self.BOUNDARY_LEFT_RIGHT = 1
        self.BOUNDARY_TOP_BOTTOM = 2

        # counter
        self.index_step = 0

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
    First:
    Calculate the curl at cell (i, j)
    This represents the vortex strength at the cell.
    Computed as: w = (del x U) where U is the velocity vector at (i, j).
    Second:
    Calculate the vorticity confinement force for each cell.
    Fvc = (N x W) where W is the curl at (i, j) and N = del |W| / |del |W||.
    N is the vector pointing to the vortex center, hence we
    add force perpendicular to N.
    """
    def vorticityConfinement(self, vcX, vcY):
        # Calculate curl(i, j) for each cell
        duDy = (self.u[1:self.h+1, 2:self.w+2] - self.u[1:self.h+1, 0:self.w]) * 0.5
        dvDx = (self.v[2:self.h+2, 1:self.w+1] - self.v[0:self.h, 1:self.w+1]) * 0.5

        v = duDy - dvDx
        
        self.curlData[1:self.h+1, 1:self.w+1] = np.abs(v)

        dx = (self.curlData[2:self.h+2, 1:self.w+1] - self.curlData[0:self.h, 1:self.w+1]) * 0.5
        dy = (self.curlData[1:self.h+1, 2:self.w+2] - self.curlData[1:self.h+1, 0:self.w]) * 0.5
        
        norm = np.sqrt(dx*dx + dy*dy)
        norm[norm == 0] = 1
        
        dx /= norm
        dy /= norm
        
        vcX[1:self.h+1, 1:self.w+1] = dy * v * -1
        vcY[1:self.h+1, 1:self.w+1] = dx * v

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
        buoy[1:self.h+1, 1:self.w+1] = a * self.d[1:self.h+1, 1:self.w+1] - b * (self.d[1:self.h+1, 1:self.w+1] - tAmb)


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
        Y, X = np.meshgrid(np.arange(1, self.w+1, 1), np.arange(1, self.h+1, 1))
        
        x = X - self.dt * self.h * u[1:self.h+1, 1:self.w+1]
        y = Y - self.dt * self.w * v[1:self.h+1, 1:self.w+1]
        
        x[x < 0.5] = 0.5
        x[x > self.h + 0.5] = self.h + 0.5
        
        i0 = x.astype(int)
        i1 = i0 + 1
        
        y[y < 0.5] = 0.5
        y[y > self.w + 0.5] = self.w + 0.5
        
        j0 = y.astype(int)
        j1 = j0 + 1
        
        s1 = x - i0
        s0 = 1 - s1
        
        t1 = y - j0
        t0 = 1 - t1
        
        d[1:self.h+1, 1:self.w+1] = s0*(t0*d0[i0, j0]+t1*d0[i0, j1]) + s1*(t0*d0[i1, j0]+t1*d0[i1, j1])
        
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
        div[1:self.h+1, 1:self.w+1] = -0.5 * ((u[2:self.h+2, 1:self.w+1] - u[0:self.h, 1:self.w+1])/self.h + (v[1:self.h+1, 2:self.w+2] - v[1:self.h+1, 0:self.w])/self.w)
        
        p.fill(0.0)

        self.setBoundary(self.BOUNDARY_NONE, div)
        self.setBoundary(self.BOUNDARY_NONE, p)

        # Solve the Poisson equations
        self.linearSolve(self.BOUNDARY_NONE, p, div, 1, 4)

        # Subtract the gradient field from the velocity field to get a mass conserving velocity field.
        u[1:self.h+1, 1:self.w+1] -= 0.5 * (p[2:self.h+2, 1:self.w+1] - p[0:self.h, 1:self.w+1]) * self.h
        v[1:self.h+1, 1:self.w+1] -= 0.5 * (p[1:self.h+1, 2:self.w+2] - p[1:self.h+1, 0:self.w]) * self.w

        self.setBoundary(self.BOUNDARY_LEFT_RIGHT, u)
        self.setBoundary(self.BOUNDARY_TOP_BOTTOM, v)

    """
    Solve a linear system of equations using Gauss-Seidel method.
    """
    def linearSolve(self, b, x, x0, a, c):
        invC = 1.0 / c
        for k in range(self.iterations):
            for i in range(1, self.h+1):
                for j in range(1, self.w+1):
                    x[i, j] = (x0[i, j] + a*(x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1])) * invC
            self.setBoundary(b, x)

    """
    Set boundary conditions.
    """
    def setBoundary(self, b, x):
        if (b == self.BOUNDARY_TOP_BOTTOM):
            x[0, 1:self.w+1] = -x[1, 1:self.w+1]
            x[self.h + 1, 1:self.w+1] = -x[self.h, 1:self.w+1]
        else:
            x[0, 1:self.w+1].fill(0.0)
            x[self.h + 1, 1:self.w+1].fill(0.0)

        if (b == self.BOUNDARY_LEFT_RIGHT):
            x[1:self.h+1, 0] = -x[1:self.h+1, 1]
            x[1:self.h+1, self.w + 1] = -x[1:self.h+1, self.w]
        else:
            x[1:self.h+1, 0].fill(0.0)
            x[1:self.h+1, self.w + 1].fill(0.0)

        x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
        x[0, self.w + 1] = 0.5 * (x[1, self.w + 1] + x[0, self.w])
        x[self.h + 1, 0] = 0.5 * (x[self.h, 0] + x[self.h + 1, 1])
        x[self.h + 1, self.w + 1] = 0.5 * (x[self.h, self.w + 1] + x[self.h + 1, self.w])

    def toRadians (self, angle):
        return angle * (np.pi/ 180)

    def update(self, f):
        self.dOld[int(self.gasLocationX), int(self.gasLocationY)] = self.gasRelease
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

        self.uOld[1:self.h+1:self.acc, 1:self.w+1:self.acc] = np.cos(self.toRadians(windDirection)) * windSpeed
        self.vOld[1:self.h+1:self.acc, 1:self.w+1:self.acc] = np.sin(self.toRadians(windDirection)) * windSpeed

        dset = f.create_dataset(f"frame{self.index_step:03d}", (self.h, self.w), dtype='f', compression="gzip")
        density_map = self.d[1:self.h + 1, 1:self.w + 1]
        dset[...] = self.gasRelease * ((density_map - density_map.min()) / (density_map.max() - density_map.min()))
        self.index_step += 1
