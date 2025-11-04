// Package hypothesis provides statistical hypothesis tests.
package hypothesis

import (
	"math"

	"gonum.org/v1/gonum/stat/distuv"
)

// TTestResult contains the results of a t-test.
type TTestResult struct {
	Statistic float64 // t-statistic
	PValue    float64 // p-value
	DF        int     // degrees of freedom
}

// TTest performs a one-sample t-test.
// Tests whether the sample mean differs from the population mean mu.
func TTest(sample []float64, mu float64) TTestResult {
	n := len(sample)
	if n < 2 {
		return TTestResult{}
	}
	
	// Calculate sample mean and std
	mean := 0.0
	for _, x := range sample {
		mean += x
	}
	mean /= float64(n)
	
	variance := 0.0
	for _, x := range sample {
		diff := x - mean
		variance += diff * diff
	}
	variance /= float64(n - 1)
	std := math.Sqrt(variance)
	
	// Calculate t-statistic
	tStat := (mean - mu) / (std / math.Sqrt(float64(n)))
	
	// Calculate p-value (two-tailed)
	df := n - 1
	tDist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(df)}
	pValue := 2 * (1 - tDist.CDF(math.Abs(tStat)))
	
	return TTestResult{
		Statistic: tStat,
		PValue:    pValue,
		DF:        df,
	}
}

// TTest2Sample performs a two-sample t-test (independent samples).
// Tests whether the means of two samples differ.
// equalVar: if true, assumes equal variances (pooled test)
func TTest2Sample(sample1, sample2 []float64, equalVar bool) TTestResult {
	n1 := len(sample1)
	n2 := len(sample2)
	
	if n1 < 2 || n2 < 2 {
		return TTestResult{}
	}
	
	// Calculate means
	mean1 := 0.0
	for _, x := range sample1 {
		mean1 += x
	}
	mean1 /= float64(n1)
	
	mean2 := 0.0
	for _, x := range sample2 {
		mean2 += x
	}
	mean2 /= float64(n2)
	
	// Calculate variances
	var1 := 0.0
	for _, x := range sample1 {
		diff := x - mean1
		var1 += diff * diff
	}
	var1 /= float64(n1 - 1)
	
	var2 := 0.0
	for _, x := range sample2 {
		diff := x - mean2
		var2 += diff * diff
	}
	var2 /= float64(n2 - 1)
	
	var tStat float64
	var df int
	
	if equalVar {
		// Pooled variance
		pooledVar := ((float64(n1-1) * var1) + (float64(n2-1) * var2)) / float64(n1+n2-2)
		se := math.Sqrt(pooledVar * (1.0/float64(n1) + 1.0/float64(n2)))
		tStat = (mean1 - mean2) / se
		df = n1 + n2 - 2
	} else {
		// Welch's t-test (unequal variances)
		se := math.Sqrt(var1/float64(n1) + var2/float64(n2))
		tStat = (mean1 - mean2) / se
		
		// Welch-Satterthwaite degrees of freedom
		num := math.Pow(var1/float64(n1)+var2/float64(n2), 2)
		denom := math.Pow(var1/float64(n1), 2)/float64(n1-1) + 
			math.Pow(var2/float64(n2), 2)/float64(n2-1)
		df = int(num / denom)
	}
	
	// Calculate p-value
	tDist := distuv.StudentsT{Mu: 0, Sigma: 1, Nu: float64(df)}
	pValue := 2 * (1 - tDist.CDF(math.Abs(tStat)))
	
	return TTestResult{
		Statistic: tStat,
		PValue:    pValue,
		DF:        df,
	}
}

// TTestPaired performs a paired t-test.
// Tests whether the mean difference between paired samples is zero.
func TTestPaired(sample1, sample2 []float64) TTestResult {
	if len(sample1) != len(sample2) {
		return TTestResult{}
	}
	
	n := len(sample1)
	if n < 2 {
		return TTestResult{}
	}
	
	// Calculate differences
	diffs := make([]float64, n)
	for i := range sample1 {
		diffs[i] = sample1[i] - sample2[i]
	}
	
	// One-sample t-test on differences
	return TTest(diffs, 0)
}
