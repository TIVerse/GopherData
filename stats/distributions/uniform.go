package distributions

import (
	"gonum.org/v1/gonum/stat/distuv"
)

// Uniform represents a continuous uniform distribution.
type Uniform struct {
	Low  float64 // lower bound
	High float64 // upper bound
	dist distuv.Uniform
}

// NewUniform creates a new uniform distribution over [low, high].
func NewUniform(low, high float64) *Uniform {
	if low >= high {
		high = low + 1
	}
	return &Uniform{
		Low:  low,
		High: high,
		dist: distuv.Uniform{Min: low, Max: high},
	}
}

// PDF returns the probability density function at x.
func (u *Uniform) PDF(x float64) float64 {
	return u.dist.Prob(x)
}

// CDF returns the cumulative distribution function at x.
func (u *Uniform) CDF(x float64) float64 {
	return u.dist.CDF(x)
}

// PPF returns the percent point function (inverse CDF) at p.
func (u *Uniform) PPF(p float64) float64 {
	return u.dist.Quantile(p)
}

// Sample generates n random samples from the distribution.
func (u *Uniform) Sample(nSamples int) []float64 {
	samples := make([]float64, nSamples)
	for i := range samples {
		samples[i] = u.dist.Rand()
	}
	return samples
}

// Mean returns the mean of the distribution.
func (u *Uniform) Mean() float64 {
	return (u.Low + u.High) / 2
}

// Variance returns the variance of the distribution.
func (u *Uniform) Variance() float64 {
	range_ := u.High - u.Low
	return (range_ * range_) / 12
}
