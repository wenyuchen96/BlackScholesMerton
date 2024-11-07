import math
from scipy.stats import norm

# a black scholes merton option prcing model that value call and put options
#S: underlying stock price in USD actuals
#K: strike price in USD actuals
#T: time to maturity in years
#R: risk free rate in decimals
#sigma: volatility in decimals


class BlackScholesMerton:
    def __init__(self, S, K, T, R, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.R = R
        self.sigma = sigma
    
    def d1_d2(self):
        d1 = (math.log(self.S / self.K) + (self.R + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return d1, d2

    def value_call(self):
        d1, d2 = self.d1_d2()
        value_call = self.S * norm.cdf(d1) - self.K * math.exp(-self.R * self.T) * norm.cdf(d2)
        return value_call
    
    def greeks_call(self):
        d1, d2 = self.d1_d2()

        delta_call = norm.cdf(d1)
        gamma_call = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))
        theta_call = (-self.S * norm.pdf(d1) * self.sigma / (2 * math.sqrt(self.T)) - self.R * self.K * math.exp(-self.R * self.T) * norm.cdf(d2))
        vega_call = self.S * norm.pdf(d1) * math.sqrt(self.T)
        rho_call = self.K * self.T * math.exp(-self.R * self.T) * norm.cdf(d2)

        return delta_call, gamma_call, theta_call, vega_call, rho_call
    
    def value_put(self):
        d1, d2 = self.d1_d2()
        value_put = self.K * math.exp(-self.R * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return value_put
    
    def greeks_put(self):
        d1, d2 = self.d1_d2()
        
        delta_put = norm.cdf(d1) - 1
        gamma_put = norm.pdf(d1) / (self.S * self.sigma * math.sqrt(self.T))
        theta_put = (-self.S * norm.pdf(d1) * self.sigma / (2 * math.sqrt(self.T)) + self.R * self.K * math.exp(-self.R * self.T) * norm.cdf(-d2))
        vega_put = self.S * norm.pdf(d1) * math.sqrt(self.T)
        rho_put = -self.K * self.T * math.exp(-self.R * self.T) * norm.cdf(-d2)

        return delta_put, gamma_put, theta_put, vega_put, rho_put
