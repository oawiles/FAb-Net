class RunningStats:

    def __init__(self, alpha):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

        self.alpha = alpha

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m * (1 - self.alpha) + self.alpha * x
            delta = x - self.old_m 
            self.new_m = self.old_m + self.alpha * delta
            self.new_s = (1 - self.alpha) * (self.old_m + self.alpha * delta * delta)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())