// Package distributions provides probability distributions.
package distributions

import (
	"gonum.org/v1/gonum/stat/distuv"
)

// Normal represents a normal (Gaussian) distribution.
type Normal struct {
	Mean float64 // mean (μ)
	Std  float64 // standard deviation (σ)
	dist distuv.Normal
}

// NewNormal creates a new normal distribution.
func NewNormal(mean, std float64) *Normal {
	if std <= 0 {
		std = 1.0
	}
	return &Normal{
		Mean: mean,
		Std:  std,
		dist: distuv.Normal{Mu: mean, Sigma: std},
	}
}

// PDF returns the probability density function at x.
func (n *Normal) PDF(x float64) float64 {
	return n.dist.Prob(x)
}

// CDF returns the cumulative distribution function at x.
// P(X <= x)
func (n *Normal) CDF(x float64) float64 {
	return n.dist.CDF(x)
}

// PPF returns the percent point function (inverse CDF) at p.
// Also known as the quantile function.
func (n *Normal) PPF(p float64) float64 {
	return n.dist.Quantile(p)
}

// Sample generates n random samples from the distribution.
func (n *Normal) Sample(nSamples int) []float64 {
	samples := make([]float64, nSamples)
	for i := range samples {
		samples[i] = n.dist.Rand()
	}
	return samples
}

// Mean returns the mean of the distribution.
func (n *Normal) MeanValue() float64 {
	return n.Mean
}

// Variance returns the variance of the distribution.
func (n *Normal) Variance() float64 {
	return n.Std * n.Std
}

// StandardNormal creates a standard normal distribution (mean=0, std=1).
func StandardNormal() *Normal {
	return NewNormal(0, 1)
}
