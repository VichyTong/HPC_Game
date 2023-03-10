import taichi as ti
import numpy as np


@ti.data_oriented
class CGPoissonSolver:
    def __init__(self, n=256, eps=1e-6, quiet=False):
        self.N = n
        self.eps = eps        
        self.quiet = quiet        
        self.real = ti.f64

        self.N_ext = 1
        self.N_tot = self.N + 2 * self.N_ext        
        self.steps = self.N * self.N  # Cg should converge within the size of the vector

        self.r = ti.field(dtype=self.real)  # residual
        self.b = ti.field(dtype=self.real)  # rhs
        self.x = ti.field(dtype=self.real)  # solution
        self.p = ti.field(dtype=self.real)  # conjugate gradient
        self.Ap = ti.field(dtype=self.real)  # matrix-vector product
        self.Ax = ti.field(dtype=self.real)  # matrix-vector product
        self.alpha = ti.field(dtype=self.real)
        self.beta = ti.field(dtype=self.real)
        ti.root.place(self.alpha, self.beta)
        ti.root.dense(ti.ij, (self.N_tot, self.N_tot)).place(self.x, self.p, self.Ap, self.r, self.Ax, self.b)


    @ti.kernel
    def init_coef(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.Ap[i, j] = 0.0
            self.Ax[i, j] = 0.0
            self.p[i, j] = 0.0
            self.x[i, j] = 0.0


    @ti.kernel
    def init_b(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.r[i, j] = ti.random()  # Random float in [0, 1)
            self.b[i, j] = self.r[i, j]


    def read_b(self):
        filename = f'./ans/b-{self.N}.npy'
        try:
            bnp = np.load(filename)
            print(f'>>> Data successfully loaded from ./ans/b-{self.N}.npy')
        except:
            print('*** Error occured during data file reading... Exiting.')
            exit()
        assert self.b.shape == bnp.shape, '*** The shape of b read from file is not consistent with problem size.'
        self.b.from_numpy(bnp)
        self.r.from_numpy(bnp)
            

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()) -> ti.f64:
        sum = 0.0
        for I in ti.grouped(p):
            sum += p[I] * q[I]
        return sum


    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.Ap[i, j] = 4.0 * self.p[i, j] - self.p[
                i + 1, j] - self.p[i - 1, j] - self.p[i, j + 1] - self.p[i,
                                                                         j - 1]
            

    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]


    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[I] -= self.alpha[None] * self.Ap[I]


    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.r[I] + self.beta[None] * self.p[I]


    def solve(self):
        initial_rTr = self.reduce(self.r, self.r)  # Compute initial residual
        if not self.quiet:
            print('>>> Initial residual =', ti.sqrt(initial_rTr))
        old_rTr = initial_rTr
        self.update_p()  # Initial p = r + beta * p ( beta = 0 )
        # -- Main loop --
        for i in range(self.steps):
            if i > 150:
                break
            self.compute_Ap()
            pAp = self.reduce(self.p, self.Ap)
            self.alpha[None] = old_rTr / pAp
            self.update_x()
            self.update_r()
            new_rTr = self.reduce(self.r, self.r)
            xTx = self.reduce(self.x, self.x)
            if ti.sqrt(new_rTr) < self.eps:
                print('>>> Conjugate Gradient method converged.')
                break
            self.beta[None] = new_rTr / old_rTr
            self.update_p()
            pTp = self.reduce(self.p, self.p)
            old_rTr = new_rTr
            if self.quiet:
                print(f'>>> Iter = {i+1:4}, Residual = {ti.sqrt(new_rTr):e}')
                print(f'>>> Iter = {i+1:4}, xTx = {ti.sqrt(xTx):e}')
                print(f'>>> Iter = {i+1:4}, pTp = {ti.sqrt(pTp):e}')

    @ti.kernel
    def check_solution(self)->ti.f64:
        residual = 0.0
        for i, j in ti.ndrange((self.N_ext, self.N_tot - self.N_ext),
                               (self.N_ext, self.N_tot - self.N_ext)):
            self.Ax[i, j] = 4.0 * self.x[i, j] - self.x[i + 1, j] - self.x[i - 1, j] - self.x[i, j + 1] - self.x[i, j - 1]
            residual += (self.b[i,j] - self.Ax[i,j]) ** 2
        return ti.sqrt(residual)

