package hypothesis

import (
	"gonum.org/v1/gonum/stat/distuv"
)

// ANOVAResult contains the results of an ANOVA test.
type ANOVAResult struct {
	FStatistic float64 // F-statistic
	PValue     float64 // p-value
	DFBetween  int     // degrees of freedom between groups
	DFWithin   int     // degrees of freedom within groups
}

// OneWayANOVA performs a one-way ANOVA test.
// Tests whether the means of multiple groups differ.
func OneWayANOVA(groups ...[]float64) ANOVAResult {
	if len(groups) < 2 {
		return ANOVAResult{}
	}
	
	k := len(groups) // number of groups
	n := 0           // total number of observations
	
	// Calculate group means and overall mean
	groupMeans := make([]float64, k)
	groupSizes := make([]int, k)
	overallSum := 0.0
	
	for i, group := range groups {
		groupSizes[i] = len(group)
		n += len(group)
		
		sum := 0.0
		for _, x := range group {
			sum += x
			overallSum += x
		}
		if len(group) > 0 {
			groupMeans[i] = sum / float64(len(group))
		}
	}
	
	if n == 0 {
		return ANOVAResult{}
	}
	
	overallMean := overallSum / float64(n)
	
	// Calculate sum of squares between groups (SSB)
	ssb := 0.0
	for i, group := range groups {
		if len(group) > 0 {
			diff := groupMeans[i] - overallMean
			ssb += float64(len(group)) * diff * diff
		}
	}
	
	// Calculate sum of squares within groups (SSW)
	ssw := 0.0
	for i, group := range groups {
		for _, x := range group {
			diff := x - groupMeans[i]
			ssw += diff * diff
		}
	}
	
	// Degrees of freedom
	dfBetween := k - 1
	dfWithin := n - k
	
	if dfBetween == 0 || dfWithin == 0 {
		return ANOVAResult{
			DFBetween: dfBetween,
			DFWithin:  dfWithin,
		}
	}
	
	// Calculate mean squares
	msb := ssb / float64(dfBetween)
	msw := ssw / float64(dfWithin)
	
	// Calculate F-statistic
	fStat := 0.0
	if msw > 0 {
		fStat = msb / msw
	}
	
	// Calculate p-value
	fDist := distuv.F{D1: float64(dfBetween), D2: float64(dfWithin)}
	pValue := 1 - fDist.CDF(fStat)
	
	return ANOVAResult{
		FStatistic: fStat,
		PValue:     pValue,
		DFBetween:  dfBetween,
		DFWithin:   dfWithin,
	}
}
