import numpy as np
import random


class FluidSimulation:
    def __init__(self,
                 n=128,
                 windDirection=0.0,
                 windLocations=32,
                 windSpeed=0.04,
                 windNoise=10,
                 windNoiseTimestep=300,
                 windLocalNoise=0.001,
                 lat=None,
                 lon=None,
                 time_points=0,
                 wind_dir=None,
                 wind_sp=None,
                 real_experiments=False,
                 simulated_experiments=True,
                 gasRelease=5500,
                 gasLocationX=64,
                 gasLocationY=64,
                 realWindSpeedNoise=0.2,
                 realWindDirectionNoise=0.2,
                 acc_ratio=None,
                 diffusion=0.0001,
                 viscosity=0):
        self.n = n
        self.NUM_OF_CELLS = n   # Number of cells (not including the boundary)
        self.VIEW_SIZE = 640    # View size (square)
        self.FPS = 60           # Frames per second

        self.CELL_SIZE = self.VIEW_SIZE / self.NUM_OF_CELLS  # Size of each cell in pixels
        self.CELL_SIZE_CEIL = np.ceil(self.CELL_SIZE) # Size of each cell in pixels (ceiling)

        self.i = np.arange(1, self.n+1)
        self.j = np.arange(1, self.n+1)
        ii, jj = np.meshgrid(self.i, self.j)
        self.ii = ii.ravel()
        self.jj = jj.ravel()
        self.Iij = self.I(self.ii, self.jj)
        self.Iij0 = self.I(self.ii + 1, self.jj)
        self.Iij1 = self.I(self.ii - 1, self.jj)
        self.Iij2 = self.I(self.ii, self.jj + 1)
        self.Iij3 = self.I(self.ii, self.jj - 1)

        self.dt = 0.1 # The simulation time-step
        self.diffusion = diffusion # The amount of diffusion
        self.viscosity = viscosity # The fluid's viscosity

        # Number of iterations to use in the Gauss-Seidel method in linearSolve()
        self.iterations = 10

        self.doVorticityConfinement = True
        self.doBuoyancy = False

        # Two extra cells in each dimension for the boundaries
        self.numOfCells = (n + 2) * (n + 2)

        self.tmp = None # Scratch space for references swapping

        self.windSpeed = windSpeed
        self.windDirection = windDirection
        #self.windDirection = np.random.rand()*360
        self.windLocations = windLocations
        # Change wind direction after N-th timesteps
        self.windNoise = windNoise
        self.windNoiseTimestep = windNoiseTimestep
        self.windNoiseCurrentStep = self.windNoiseTimestep
        self.windNoiseCurrent = 0
        # Local noise at each location
        self.windLocalNoise = windLocalNoise
        # Real wind noise
        self.realWindSpeedNoise = realWindSpeedNoise
        self.realWindDirectionNoise = realWindDirectionNoise

        self.gasRelease = gasRelease
        self.gasLocationX = int(gasLocationX)
        self.gasLocationY = int(gasLocationY)
        # Self might benefit from using typed arrays like Float32Array in some configuration.
        # But I haven't seen any significant improvement on Chrome because V8 probably does it on its own.

        # Values for current simulation step, and initialize everything to zero
        self.u = np.zeros(self.numOfCells) * 0.001 # Velocity x
        self.v = np.zeros(self.numOfCells) * 0.001 # Velocity y
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
        
        self.index = 0

        #Extra information added for experiments
        self.lat = lat
        self.lon = lon
        if (simulated_experiments):
            self.time_points = time_points
        else:
            self.time_points = time_points/self.dt
        self.wind_dir = wind_dir
        self.wind_sp = wind_sp
        self.current_timestep = 0
        self.real_experiments = real_experiments
        self.simulated_experiments = simulated_experiments
        self.acc_ratio = acc_ratio
        self.active_step = 0
        self.f = None

    def randomWind(self):
        return (random.random()*2-1)*self.windLocalNoise

    def I(self, i, j):
        return i + (self.n + 2) * j

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
        tAmb = np.sum(self.d) / (self.n * self.n)

        # For each cell compute buoyancy force
        buoy[self.Iij] = a * self.d[self.Iij] - b * (self.d[self.Iij] - tAmb)


    """
    Diffuse the density between neighbouring cells.
    """
    def diffuse(self, b, x, x0, diffusion):
        a = self.dt * diffusion * self.n * self.n

        self.linearSolve(b, x, x0, a, 1 + 4 * a)

    """
    The advection step moves the density through the static velocity field.
    Instead of moving the cells forward in time, we treat the cell's center as a particle
    and then trace it back in time to look for the 'particles' which end up at the cell's center.
    """
    def advect(self, b, d, d0, u, v):
        dt0 = self.dt * self.n

        x = self.ii - dt0 * u[self.Iij]
        y = self.jj - dt0 * v[self.Iij]
        
        x[x < 0.5] = 0.5
        x[x > self.n + 0.5] = self.n + 0.5
        
        i0 = x.astype(int)
        i1 = i0 + 1
        
        y[y < 0.5] = 0.5
        y[y > self.n + 0.5] = self.n + 0.5
        
        j0 = y.astype(int)
        j1 = j0 + 1
        
        s1 = x - i0
        s0 = 1 - s1
        
        t1 = y - j0
        t0 = 1 - t1
        
        Iij0 = self.I(i0, j0)
        Iij1 = self.I(i0, j1)
        Iij2 = self.I(i1, j0)
        Iij3 = self.I(i1, j1)
        
        d[self.Iij] = s0 * (t0 * d0[Iij0] + t1 * d0[Iij1]) + s1 * (t0 * d0[Iij2] + t1 * d0[Iij3])

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
        h = 1.0 / self.n
        
        div[self.Iij] = -0.5 * h * (u[self.Iij0] - u[self.Iij1] + v[self.Iij2] - v[self.Iij3])
        
        p.fill(0.0)

        self.setBoundary(self.BOUNDARY_NONE, div)
        self.setBoundary(self.BOUNDARY_NONE, p)

        # Solve the Poisson equations
        self.linearSolve(self.BOUNDARY_NONE, p, div, 1, 4)

        # Subtract the gradient field from the velocity field to get a mass conserving velocity field.
        u[self.Iij] -= 0.5 * (p[self.Iij0] - p[self.Iij1]) / h
        v[self.Iij] -= 0.5 * (p[self.Iij2] - p[self.Iij3]) / h

        self.setBoundary(self.BOUNDARY_LEFT_RIGHT, u)
        self.setBoundary(self.BOUNDARY_TOP_BOTTOM, v)

    """
    Solve a linear system of equations using Gauss-Seidel method.
    """
    def linearSolve(self, b, x, x0, a, c):
        invC = 1.0 / c
        for k in range(self.iterations):
            x[self.Iij] = (x0[self.Iij] + a*(x[self.Iij1] + x[self.Iij0] + x[self.Iij3] + x[self.Iij2])) * invC
            self.setBoundary(b, x)

    """
    Set boundary conditions.
    """
    def setBoundary(self, b, x):
        if (b == self.BOUNDARY_LEFT_RIGHT):
            x[self.I(0, self.j)] = -x[self.I(1, self.j)]
            x[self.I(self.n + 1, self.j)] = -x[self.I(self.n, self.j)]
        else:
            x[self.I(0, self.j)].fill(0.0)
            x[self.I(self.n + 1, self.j)].fill(0.0)

        if (b == self.BOUNDARY_TOP_BOTTOM):
            x[self.I(self.i, 0)] = -x[self.I(self.i, 1)]
            x[self.I(self.i, self.n + 1)] = -x[self.I(self.i, self.n)]
        else:
            x[self.I(self.i, 0)].fill(0.0)
            x[self.I(self.i, self.n + 1)].fill(0.0)

        x[self.I(0, 0)] = 0.5 * (x[self.I(1, 0)] + x[self.I(0, 1)])
        x[self.I(0, self.n + 1)] = 0.5 * (x[self.I(1, self.n + 1)] + x[self.I(0, self.n)])
        x[self.I(self.n + 1, 0)] = 0.5 * (x[self.I(self.n, 0)] + x[self.I(self.n + 1, 1)])
        x[self.I(self.n + 1, self.n + 1)] = 0.5 * (x[self.I(self.n, self.n + 1)] + x[self.I(self.n + 1, self.n)])

    def toRadians (self, angle):
        return angle * (np.pi/ 180)

    def update(self, f):
        self.dOld[int(self.I(self.gasLocationX, self.gasLocationY))] = self.gasRelease
        # Step the fluid simulation
        self.velocityStep()
        self.densityStep()
        if (self.real_experiments==True):
            earlier_timesteps = self.time_points[self.time_points<self.current_timestep]
            latest_ealier = len(earlier_timesteps)
            if (self.simulated_experiments):
                self.windDirection = self.wind_dir[latest_ealier]
            else:
                self.windDirection = np.rad2deg(self.wind_dir[latest_ealier])
            if (self.simulated_experiments):
                self.windSpeed = self.wind_sp[latest_ealier]
            else:
                self.windSpeed = self.wind_sp[latest_ealier]*self.n/100000
            self.current_timestep += 1
            self.activeDirectionNoise = random.random()*2*self.realWindDirectionNoise-self.realWindDirectionNoise
            self.windDirection += self.activeDirectionNoise*360
            self.activeSpeedNoise = random.random()*2*self.realWindSpeedNoise-self.realWindSpeedNoise
            self.windSpeed += self.activeSpeedNoise*self.windSpeed
            rand = 0
        else:
            # Artificial wind field
            if self.windNoiseCurrentStep == self.windNoiseTimestep:
                    rand = (np.random.rand()*2 - 1)*self.windNoise
                    self.windNoiseCurrent = rand
                    self.windNoiseCurrentStep = 0
            else:
                    rand = self.windNoiseCurrent
                    self.windNoiseCurrentStep += 1
        #print (self.windDirection)
        du = np.sin(self.toRadians(self.windDirection + rand))*self.windSpeed
        dv = np.cos(self.toRadians(self.windDirection + rand))*self.windSpeed

        acc = self.NUM_OF_CELLS / self.windLocations
        jj, ii = np.meshgrid(np.arange(0, self.NUM_OF_CELLS, int(acc)), np.arange(0, self.NUM_OF_CELLS, int(acc)))
        self.uOld[self.I(ii.ravel(), jj.ravel())] = du if self.real_experiments else du + self.randomWind()
        self.vOld[self.I(ii.ravel(), jj.ravel())] = dv if self.real_experiments else dv + self.randomWind()

        dset = f.create_dataset('frame' + str(self.index), (self.n, self.n), dtype='f', compression="gzip")
        density_map = self.d.reshape(self.n + 2, self.n + 2)[1:self.n + 1, 1:self.n + 1]
        dset[...] = self.gasRelease * ((density_map - density_map.min()) / (density_map.max() - density_map.min()))
        self.index += 1
        self.active_step +=1