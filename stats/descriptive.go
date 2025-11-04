// Package stats provides statistical functions and algorithms.
package stats

import (
	"math"
	"sort"
)

// DescriptiveStats contains summary statistics for a dataset.
type DescriptiveStats struct {
	Count    int
	Mean     float64
	Std      float64
	Min      float64
	Q25      float64
	Median   float64
	Q75      float64
	Max      float64
	Skewness float64
	Kurtosis float64
}

// Mean calculates the arithmetic mean of values.
func Mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// Median calculates the median of values.
func Median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	n := len(sorted)
	if n%2 == 0 {
		return (sorted[n/2-1] + sorted[n/2]) / 2
	}
	return sorted[n/2]
}

// Mode finds the most frequent value and its frequency.
func Mode(values []float64) (float64, int) {
	if len(values) == 0 {
		return 0, 0
	}
	
	freq := make(map[float64]int)
	for _, v := range values {
		freq[v]++
	}
	
	maxFreq := 0
	mode := 0.0
	for val, count := range freq {
		if count > maxFreq {
			maxFreq = count
			mode = val
		}
	}
	
	return mode, maxFreq
}

// Std calculates the sample standard deviation.
func Std(values []float64) float64 {
	return math.Sqrt(Var(values))
}

// Var calculates the sample variance.
func Var(values []float64) float64 {
	n := len(values)
	if n < 2 {
		return 0
	}
	
	mean := Mean(values)
	sumSq := 0.0
	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}
	
	return sumSq / float64(n-1) // Sample variance (Bessel's correction)
}

// Range calculates the range (max - min) of values.
func Range(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	min := values[0]
	max := values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	
	return max - min
}

// IQR calculates the interquartile range (Q3 - Q1).
func IQR(values []float64) float64 {
	q1 := Quantile(values, 0.25)
	q3 := Quantile(values, 0.75)
	return q3 - q1
}

// Skew calculates the sample skewness (Fisher-Pearson coefficient).
func Skew(values []float64) float64 {
	n := len(values)
	if n < 3 {
		return 0
	}
	
	mean := Mean(values)
	std := Std(values)
	if std == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/std, 3)
	}
	
	// Sample skewness with bias correction
	return (float64(n) / (float64(n-1) * float64(n-2))) * sum
}

// Kurtosis calculates the excess kurtosis.
func Kurtosis(values []float64) float64 {
	n := len(values)
	if n < 4 {
		return 0
	}
	
	mean := Mean(values)
	std := Std(values)
	if std == 0 {
		return 0
	}
	
	sum := 0.0
	for _, v := range values {
		sum += math.Pow((v-mean)/std, 4)
	}
	
	// Excess kurtosis with bias correction
	kurt := (float64(n*(n+1)) / (float64((n-1)*(n-2)*(n-3)))) * sum
	correction := 3 * float64((n-1)*(n-1)) / float64((n-2)*(n-3))
	return kurt - correction
}

// Quantile calculates the q-th quantile (0 <= q <= 1).
// Uses linear interpolation between closest ranks.
func Quantile(values []float64, q float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if q < 0 {
		q = 0
	}
	if q > 1 {
		q = 1
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	pos := q * float64(len(sorted)-1)
	lower := int(math.Floor(pos))
	upper := int(math.Ceil(pos))
	
	if lower == upper {
		return sorted[lower]
	}
	
	// Linear interpolation
	weight := pos - float64(lower)
	return sorted[lower]*(1-weight) + sorted[upper]*weight
}

// Percentile calculates the p-th percentile (0 <= p <= 100).
func Percentile(values []float64, p float64) float64 {
	return Quantile(values, p/100.0)
}

// Describe returns comprehensive descriptive statistics.
func Describe(values []float64) DescriptiveStats {
	if len(values) == 0 {
		return DescriptiveStats{}
	}
	
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)
	
	return DescriptiveStats{
		Count:    len(values),
		Mean:     Mean(values),
		Std:      Std(values),
		Min:      sorted[0],
		Q25:      Quantile(values, 0.25),
		Median:   Median(values),
		Q75:      Quantile(values, 0.75),
		Max:      sorted[len(sorted)-1],
		Skewness: Skew(values),
		Kurtosis: Kurtosis(values),
	}
}
