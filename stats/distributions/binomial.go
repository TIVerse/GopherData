package distributions

import (
	"math"
	"math/rand"
	"time"
)

// Binomial represents a binomial distribution.
type Binomial struct {
	N float64 // number of trials
	P float64 // success probability
}

// NewBinomial creates a new binomial distribution.
func NewBinomial(n int, p float64) *Binomial {
	if n < 0 {
		n = 1
	}
	if p < 0 {
		p = 0
	}
	if p > 1 {
		p = 1
	}
	return &Binomial{
		N: float64(n),
		P: p,
	}
}

// PMF returns the probability mass function at k.
// P(X = k)
func (b *Binomial) PMF(k int) float64 {
	if k < 0 || float64(k) > b.N {
		return 0
	}
	
	// PMF = C(n,k) * p^k * (1-p)^(n-k)
	// where C(n,k) is the binomial coefficient
	
	coeff := binomialCoefficient(int(b.N), k)
	prob := coeff * math.Pow(b.P, float64(k)) * math.Pow(1-b.P, b.N-float64(k))
	
	return prob
}

// CDF returns the cumulative distribution function at k.
// P(X <= k)
func (b *Binomial) CDF(k int) float64 {
	if k < 0 {
		return 0
	}
	if float64(k) >= b.N {
		return 1
	}
	
	cdf := 0.0
	for i := 0; i <= k; i++ {
		cdf += b.PMF(i)
	}
	
	return cdf
}

// Sample generates n random samples from the distribution.
func (b *Binomial) Sample(nSamples int) []int {
	samples := make([]int, nSamples)
	
	for i := range samples {
		successes := 0
		for trial := 0; trial < int(b.N); trial++ {
			if randomFloat() < b.P {
				successes++
			}
		}
		samples[i] = successes
	}
	
	return samples
}

// Mean returns the mean of the distribution.
func (b *Binomial) Mean() float64 {
	return b.N * b.P
}

// Variance returns the variance of the distribution.
func (b *Binomial) Variance() float64 {
	return b.N * b.P * (1 - b.P)
}

// Helper functions

// binomialCoefficient calculates C(n, k) = n! / (k! * (n-k)!)
func binomialCoefficient(n, k int) float64 {
	if k > n || k < 0 {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	
	// Use the more efficient formula: C(n,k) = C(n,k-1) * (n-k+1) / k
	// But for simplicity, use the direct calculation with logarithms for large values
	
	if k > n-k {
		k = n - k
	}
	
	result := 1.0
	for i := 0; i < k; i++ {
		result *= float64(n - i)
		result /= float64(i + 1)
	}
	
	return result
}

// randomFloat generates a random float between 0 and 1 using math/rand
var rng = rand.New(rand.NewSource(time.Now().UnixNano()))

func randomFloat() float64 {
	return rng.Float64()
}

// SetSeed sets the random seed for reproducible results
func SetSeed(seed int64) {
	rng = rand.New(rand.NewSource(seed))
}
