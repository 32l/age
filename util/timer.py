import time

class Timer():
	def __init__(self):
		self._time = 0

	def tic(self):
		self._time = time.time()

	def toc(self):
		t = time.time() - self._time
		# print(t)
		return t