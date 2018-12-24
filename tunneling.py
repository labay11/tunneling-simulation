#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python 2.x and 3.x compatibility
# NOTE: if using python 3.x change: xrange --> range
from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

current_milli_time = lambda: int(round(time.time() * 1000))

def format_float_to_filename(x, tmpl):
	a = tmpl % x
	return a.replace('.', '_')

def get_filename(tmp, V0, L, l):
	a = '%.2f_%.2f_%.2f' % (V0, L, l)
	return tmp % a.replace('.', '_')

FILE_ENERGIES = 'energies/energies_%s.txt'
FILE_PHI = 'phis/phi_%s.txt'
FILE_TIMES = 'times/times_%s.txt'

if not os.path.exists('energies'):
	os.makedirs('energies')
if not os.path.exists('phis'):
	os.makedirs('phis')
if not os.path.exists('times'):
    os.makedirs('times')

k = 5.12665			  	# sqrt(2m)/hbar (  eV^(-1/2)*nm^(-1)  )
hbar = 6.582e-4			# h barra (eV · ps)

class Tunneling(object):

	def __init__(self, V0, L, l, xi, sigmax, T, Ne=256, dE=0.0001, dx=0.01):
		'''
			Tunneling class to simulate experiments on the tunneling effect
			due to a potential barrier of width 2l located at the center
			of an infinite square box of width 2L.

			:float V0: height f the potential barrier in eV
			:float L: half length of the box in nm
			:float l: half length of the potentil barrier in nm
			:float xi: initial position of the gaussian wave packed,
						must be between [-L, L]
			:float sigmax: width of the gaussian wave packet
			:float T: energy of the kick in eV
			:int Ne: number of energies to consider
			:float dE: interval between energies
			:float dx: space step
		'''
		self.V0 = V0

		self.L = L
		self.l = l
		self.xi = xi
		self.sigmax = sigmax

		self.T = T
		self.Ne = Ne 
		self.dE = dE 

		self.dx = dx
		self.Nx = int(2 * self.L / self.dx)
		self.Nx1 = int((self.L - self.l) / self.dx)
		self.Nx2 = int(2 * self.l / self.dx)
		self.Nx3 = self.Nx1
		self.X = np.linspace(-self.L, self.L, self.Nx)

		self.norm = None

	def print_info(self):
		print('V0 = %.2f, T = %.2f, L = %.2f, l = %.2f' % (self.V0, self.T, self.L, self.l))

	#
	# Transcendental equations for the even and odd cases
	# and the different energies:
	#		 E < V0 --> *_l
	#		 E > V0 --> *_g
	#
	def _even_g(self, E):
		return np.sqrt(E-self.V0)*np.sin(k*np.sqrt(E-self.V0)*self.l)*np.sin(k*np.sqrt(E)*(self.L-self.l)) - \
			np.sqrt(E)*np.cos(k*(np.sqrt(E-self.V0))*self.l)*np.cos(k*np.sqrt(E)*(self.L-self.l))

	def _even_l(self, E):
		return np.sqrt(self.V0-E)*np.sinh(k*np.sqrt(self.V0-E)*self.l)*np.sin(k*np.sqrt(E)*(self.L-self.l)) + \
			np.sqrt(E)*np.cosh(k*(np.sqrt(self.V0-E))*self.l)*np.cos(k*np.sqrt(E)*(self.L-self.l))

	def _odd_l(self, E):
		return np.sqrt(self.V0-E)*np.cosh(k*(np.sqrt(self.V0-E))*self.l)*np.sin(k*np.sqrt(E)*(self.L-self.l)) + \
			np.sqrt(E)*np.sinh(k*(np.sqrt(self.V0-E))*self.l)*np.cos(k*np.sqrt(E)*(self.L-self.l))

	def _odd_g(self, E):
		return np.sqrt(E-self.V0)*np.cos(k*np.sqrt(E-self.V0)*self.l)*np.sin(k*np.sqrt(E)*(self.L-self.l)) + \
			np.sqrt(E)*np.sin(k*np.sqrt(E-self.V0)*self.l)*np.cos(k*np.sqrt(E)*(self.L-self.l))

	def _even_n(self, E):
		return np.sqrt(E - self.V0)*np.sin(k*np.sqrt(E-self.V0)*self.l)*np.tanh(k*np.sqrt(-E)*(self.L-self.l)) + \
			np.sqrt(-E)*np.cos(k*np.sqrt(E-self.V0)*self.l)

	def _odd_n(self, E):
		return np.sqrt(E - self.V0)*np.cos(k*np.sqrt(E-self.V0)*self.l)*np.tanh(k*np.sqrt(-E)*(self.L-self.l)) - \
			np.sqrt(-E)*np.sin(k*np.sqrt(E-self.V0)*self.l)

	def find_energies(self):
		# not very optimized, almost 30s to complete on a good computer
		E = 0 if self.V0 >= 0 else self.V0

		Ep = [] # energy of the states
		i = 0 # number of states

		last_even, last_odd = 0, 0

		print('Start root finding...', end=' ')
		start = current_milli_time()

		while i < self.Ne:
			if E < 0:
				#print('Entered in n', i)
				e, o = self._even_n(E), self._odd_n(E)
			elif E < self.V0:
				#print('Entered in l', i)
				e, o = self._even_l(E), self._odd_l(E)
			else:
				#print('Entered in g', i)
				e, o = self._even_g(E), self._odd_g(E)

			if e * last_even < 0: # change of sign, root found
				# approximate the root by the medium value
				Ep.append(E - self.dE/2) 	# Ep.append(E)
				i+=1

			# cannot join both if because in that case multiplicities will not be counted
			if o * last_odd < 0: 
				Ep.append(E - self.dE/2)
				i+=1

			last_even, last_odd = e, o
			E += self.dE

		print('OK (%.2f s)' % ((current_milli_time() - start) / 1000))

		return np.array(sorted(Ep)) # assume even and ood energies are intercalated

	def save_energies(self, E):
		with open(get_filename(FILE_ENERGIES, self.V0, self.L, self.l), 'w') as outf:
			for k in range(len(E)):
				outf.write('%d\t%.4g\n' % (k, E[k]))

	def read_energies(self):
		Ep = []
		with open(get_filename(FILE_ENERGIES, self.V0, self.L, self.l)) as f:
			for line in f:
				Ep.append(float(line.split('\t')[1].strip()))
		return np.array(Ep)

	#########################################################
	## Define the wave function for the even and odd cases ##
	#########################################################
	def _phi_even_l(self, reg, E, x):
		if reg == 1:
			return np.sin(k*np.sqrt(E)*(x+self.L))
		elif reg == 2:
			return np.sin(k*np.sqrt(E)*(self.L-self.l))*np.cosh(k*np.sqrt(self.V0-E)*x)/(np.cosh(k*np.sqrt(self.V0-E)*self.l))
		elif reg == 3:
			return -np.sin(k*np.sqrt(E)*(x-self.L))

	def _phi_even_g(self, reg, E, x):
		if reg == 1:
			return np.sin(k*np.sqrt(E)*(x+self.L))
		elif reg == 2:
			return np.sin(k*np.sqrt(E)*(self.L-self.l))*np.cos(k*np.sqrt(E-self.V0)*x)/(np.cos(k*np.sqrt(E-self.V0)*self.l))
		elif reg == 3:
			return -np.sin(k*np.sqrt(E)*(x-self.L))

	def _phi_odd_l(self, reg, E, x):
		if reg == 1:
			return np.sin(k*np.sqrt(E)*(x+self.L))
		elif reg == 2:
			return -np.sin(k*np.sqrt(E)*(self.L-self.l))*np.sinh(k*np.sqrt(self.V0-E)*x)/(np.sinh(k*np.sqrt(self.V0-E)*self.l))
		elif reg == 3:
			return np.sin(k*np.sqrt(E)*(x-self.L))

	def _phi_odd_g(self, reg, E, x):
		if reg == 1:
			return np.sin(k*np.sqrt(E)*(x+self.L))
		elif reg == 2:
			return -np.sin(k*np.sqrt(E)*(self.L-self.l))*np.sin(k*np.sqrt(E-self.V0)*x)/(np.sin(k*np.sqrt(E-self.V0)*self.l))
		elif reg == 3:
			return np.sin(k*np.sqrt(E)*(x-self.L))

	def _phi_even_n(self, reg, E, x):
		if reg == 1:
			return np.sinh(k*np.sqrt(-E)*(x+self.L))
		elif reg == 2:
			return np.sinh(k*np.sqrt(-E)*(self.L-self.l))*np.cos(k*np.sqrt(E-self.V0)*x)/(np.cos(k*np.sqrt(E-self.V0)*self.l))
		elif reg == 3:
			return -np.sinh(k*np.sqrt(-E)*(x-self.L))

	def _phi_odd_n(self, reg, E, x):
		if reg == 1:
			return np.sinh(k*np.sqrt(-E)*(x+self.L))
		elif reg == 2:
			return -np.sinh(k*np.sqrt(-E)*(self.L-self.l))*np.sin(k*np.sqrt(E-self.V0)*x)/(np.sin(k*np.sqrt(E-self.V0)*self.l))
		elif reg == 3:
			return np.sinh(k*np.sqrt(-E)*(x-self.L))
	
	def phi_even(self, reg, E, x):
		if E < 0:
			return self._phi_even_n(reg, E, x)
		elif E < self.V0:
			return self._phi_even_l(reg, E, x)
		else:
			return self._phi_even_g(reg, E, x)

	def phi_odd(self, reg, E, x):
		if E < 0:
			return self._phi_odd_n(reg, E, x)
		elif E < self.V0:
			return self._phi_odd_l(reg, E, x)
		else:
			return self._phi_odd_g(reg, E, x)

	def evaluate_wave_function(self, save=False):
		# wave function matrix
		PHI = np.zeros((self.Ne, self.Nx))

		# define the 3 difference regions for x
		x1, x2, x3 = np.linspace(-self.L, -self.l, self.Nx1), np.linspace(-self.l, self.l, self.Nx2), np.linspace(self.l, self.L, self.Nx3) 

		for i in range(self.Ne): # loop over states
			E = self.Ep[i] 
			if i % 2 == 0:
				PHI[i, :self.Nx1] = self.phi_even(1, E, x1)
				PHI[i, self.Nx1:self.Nx2+self.Nx1] = self.phi_even(2, E, x2)
				PHI[i, self.Nx1+self.Nx2:] = self.phi_even(3, E, x3)
			else:
				PHI[i, :self.Nx1] = self.phi_odd(1, E, x1)
				PHI[i, self.Nx1:self.Nx2+self.Nx1] = self.phi_odd(2, E, x2)
				PHI[i, self.Nx1+self.Nx2:] = self.phi_odd(3, E, x3)

			# normalise the wave function (as a discrete sum)
			PHI[i] /= np.sqrt(np.sum(PHI[i]**2) * self.dx)

		if save:
			np.savetxt(get_filename(FILE_PHI, self.V0, self.L, self.l), PHI.transpose(), fmt='%10.4f', delimiter='\t')

		return PHI

	def gaussian(self, x):
		def f(x):
			return np.exp(- (x - self.xi)**2 / (4 * self.sigmax**2))

		if self.norm == None:
			self.norm = 1.0 / np.sqrt(integrate.quad(lambda x: f(x)**2, -self.L, self.L)[0])

		return self.norm * f(x)
	
	def kick(self, func, x):
		return np.exp(1j * k * np.sqrt(self.T) * x) * func(x)

	def expand_function(self, f):
		# discrete sum as an approximation to the integral
		return np.sum(self.PHI * f, axis=1) * self.dx

	def time_evolution(self, coef, t_max, dt):
		Nt = int(t_max / dt)
		times = np.zeros((Nt, self.Nx))

		for k in range(Nt):
			t = k * dt

			times[k] = np.abs(np.dot(coef * np.exp(-1j * self.Ep * t / hbar), self.PHI))**2

		return times

	def experiment(self, t_max, dt):
		# evalutate the energies if not saved, otherwise read them from file
		if os.path.exists(get_filename(FILE_ENERGIES, self.V0, self.L, self.l)):
			self.Ep = self.read_energies()
		else:
			self.Ep = self.find_energies()
			self.save_energies(self.Ep)

		self.PHI = self.evaluate_wave_function()

		func = self.kick(self.gaussian, self.X)
		self.C0 = self.expand_function(func)

		return self.time_evolution(self.C0, t_max, dt)

	def save_times(self, times):
		np.savetxt(get_filename(FILE_TIMES, self.V0, self.L, self.l), times, fmt='%10.4f', delimiter='\t')

	def expected_value(self, f, p):
		return np.sum(p * f(self.X)) * self.dx

	def plot(self, times, T_MAX, dt, filename=None, interval=50):

		X2 = self.X**2
		Emax = 20.0

		if self.V0 < 0:
			zerox = -self.V0 / (Emax - self.V0)
		else:
			zerox = 0

		def expected_x(ts):
			return np.sum(ts * self.X) * self.dx
		def sigma_x(ts, x):
			x2 = np.sum(ts * X2) * self.dx
			return np.sqrt(x2 - x**2)

		def expected_E():
			return np.sum(np.abs(self.C0)**2 * self.Ep)
		def sigma_E(e):
			return np.sqrt(np.sum((np.abs(self.C0) * self.Ep)**2) - e**2)

		expE = expected_E()
		sigE = sigma_E(expE)
		print('<H> = %.5f +/- %.5f' % (expE, sigE))

		# scale factor for energy 
		expE = expE/Emax + zerox 
		sigE /= Emax

		Nt = int(T_MAX / dt)

		def update(t, x, times, lines):
			i = int(t / dt)
			if i >= Nt:
				i = Nt - 1
			
			e = expected_x(times[i])
			s = sigma_x(times[i], e)

			lines[0].set_data(x, times[i])
			lines[1].set_data([e, e], [expE - sigE, expE + sigE])
			lines[2].set_data([e - s, e + s], [expE, expE])

			lines[3].set_text('%.3f ps' % t)

			return lines 

		fig, ax1 = plt.subplots()
		ax1.set_xlabel(r'$x\ (nm)$')
		ax1.set_ylabel(r'$|\psi(x)|^2$')
		ax1.set_xlim(-self.L, self.L)
		ax1.set_ylim(0, 1)

		ax2 = ax1.twinx()
		ax2.set_ylim(min(0, self.V0), Emax)
		ax2.set_ylabel(r'$E\ (eV)$')

		# plot potential
		ax1.plot([-self.L, -self.L, -self.l, -self.l, self.l, self.l, self.L, self.L], [1, zerox, zerox, self.V0/Emax, self.V0/Emax, zerox, zerox, 1], c='k', lw=2)
		
		line1 = ax1.plot([], [], color='b', lw=0.8, animated=True)[0]
		line2 = ax1.plot([], [], c='r', lw=0.8)[0]
		line3 = ax1.plot([], [], c='r', lw=0.5)[0]

		text = ax1.text(-self.L + 0.02, 0.96, '', fontsize=9)

		ani = FuncAnimation(fig, update, fargs=(self.X, times, [line1, line2, line3, text]), frames=np.linspace(0, T_MAX, Nt),
		                    blit=True, interval=interval, repeat=False)

		if filename is not None:
			ani.save(filename, fps=20, writer="avconv", codec="libx264")
			print('Plot saved as', filename)
		else:
			plt.show()

def get_args():
	import argparse

	parser = argparse.ArgumentParser(description='Quantum tunneling effect.')
	parser.add_argument('V0', metavar='V0', type=float,
	                    help='height of the barrier in eV')
	parser.add_argument('L', metavar='L', type=float,
	                    help='half length of the box in nm')
	parser.add_argument('l', metavar='l', type=float,
	                    help='half length of the barrier in nm')
	parser.add_argument('T', metavar='T', type=float,
	                    help='kick in eV')
	parser.add_argument('xi', metavar='xi', type=float,
	                    help='center of the gaussian')
	parser.add_argument('sx', metavar='sigmax', type=float,
	                    help='size of the gaussian')

	parser.add_argument('--TMAX', metavar='TMAX', type=float,
	                    default=1, help='max time')
	parser.add_argument('--dt', metavar='dt', type=float,
	                    default=0.001, help='step in time')
	parser.add_argument('--filename', metavar='filename', type=str,
	                    default=None, help='animation destination file')

	return parser.parse_args()

if __name__ == '__main__':
	args = get_args()

	tun = Tunneling(args.V0, args.L, args.l, args.T, args.xi, args.sx)
	times = tun.experiment(args.TMAX, args.dt)
	tun.plot(times, args.TMAX, args.dt, args.filename)
